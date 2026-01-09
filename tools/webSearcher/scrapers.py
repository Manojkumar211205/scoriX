
import requests
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
import io
import re
import os
from dotenv import load_dotenv
import serpapi
import concurrent.futures
try:
    from keybert import KeyBERT
except ImportError:
    KeyBERT = None

load_dotenv()

# Global model cache
_kw_model = None
def get_kw_model():
    global _kw_model
    if _kw_model is None and KeyBERT is not None:
        try:
             # Use a smaller model for speed
            _kw_model = KeyBERT('all-MiniLM-L6-v2')
        except Exception as e:
            pass
    return _kw_model

# Try importing pypdf, handle if missing
try:
    import pypdf
except ImportError:
    pypdf = None

class BaseScraper(ABC):
    """
    Abstract base class for all web scrapers.
    """
    
    @abstractmethod
    def fetch(self, query: str) -> dict:
        """
        Fetches and extracts content from the given query (URL).
        
        Args:
            query (str): The URL to scrape.
            
        Returns:
            dict: A dictionary containing 'topic', 'content', and 'source'.
        """
        pass

    def _clean_text(self, text: str) -> str:
        """Helper to clean whitespace from text."""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip()

    def _resolve_url(self, query: str, site_filter: str = None) -> str:
        """
        Resolves a search query to a URL using SerpApi.
        If query is already a URL, returns it.
        """
        # Check if query is a URL
        if re.match(r'^https?://', query):
            return query

        api_key = os.getenv("SERPAPI") or os.getenv("SERPAPI_API_KEY")
        if not api_key:
            pass
            return query

        try:
            search_query = query
            if site_filter:
                search_query += f" site:{site_filter}"
            
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": api_key,
                "num": 1
            }
            
            results = serpapi.search(params)
            organic_results = results.get("organic_results", [])
            
            if organic_results:
                return organic_results[0].get("link")
            
        except Exception as e:
            pass
            
        return query

    def _extract_additional_topics(self, content: str) -> list:
        """Extracts keywords/topics from content using KeyBERT with safe chunking and multi-threading."""
        kw_model = get_kw_model()
        if not kw_model or not content:
            return []

        try:
            # -------- Chunk into ~400-word blocks --------
            words = content.split()
            chunk_size = 400
            chunks = [
                " ".join(words[i:i + chunk_size])
                for i in range(0, len(words), chunk_size)
            ]

            all_keywords = []

            def process_chunk(chunk):
                try:
                    keywords = kw_model.extract_keywords(
                        chunk,
                        keyphrase_ngram_range=(1, 2),  # 1–2 word topics
                        stop_words='english',
                        top_n=5                         # topics per chunk
                    )
                    return [kw[0] for kw in keywords]
                except Exception:
                    return []

            # Use ThreadPoolExecutor for parallel processing
            # Using threads because KeyBERT/PyTorch might release GIL for heavy ops
            max_workers = 12
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_chunk, chunks))
            
            for res in results:
                all_keywords.extend(res)

            # -------- Deduplicate while keeping order --------
            seen = set()
            unique_topics = []
            for kw in all_keywords:
                kw_lower = kw.lower()
                if kw_lower not in seen:
                    seen.add(kw_lower)
                    unique_topics.append(kw)

            return unique_topics

        except Exception as e:
            # print(f"KeyBERT extraction failed: {e}") # Keeping clean logs
            return []


    def _get_soup(self, url: str) -> BeautifulSoup:
        """Helper to get BeautifulSoup object."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')

class W3SchoolsScraper(BaseScraper):
    """
    Scraper for W3Schools.
    Scrapes definitions, syntax, and simple examples.
    Ignores ads, navigation, and unrelated sections.
    """
    def fetch(self, query: str) -> dict:
        try:
            url = self._resolve_url(query, site_filter="w3schools.com")
            soup = self._get_soup(url)
            
            # Main content area usually in 'w3-main' or 'main'
            main_content = soup.find('div', {'id': 'main'}) or soup.find('div', class_='w3-main')
            
            if not main_content:
                return {"topic": "", "content": "", "source": "W3Schools", "error": "Could not find main content"}

            # Extract Topic (H1)
            h1 = main_content.find('h1')
            topic = h1.get_text(strip=True) if h1 else ""
            if h1:
                h1.decompose() # Remove from content flow to avoid duplication

            # Cleanup
            for unwanted in main_content.find_all(['div'], class_=['w3-col', 'w3-sidebar', 'w3-bar', 'nextprev']):
                unwanted.decompose()
            for pattern in [re.compile(r'ad'), re.compile(r'advert')]:
                for ad in main_content.find_all(class_=pattern):
                    ad.decompose()

            extracted_parts = []
            
            # We want definitions (h2 + p), syntax (pre/code), simple examples (div.w3-example)
            for element in main_content.find_all(['h2', 'p', 'pre', 'div']):
                if element.name == 'div' and 'w3-example' in element.get('class', []):
                    # Handle example
                    code_box = element.find(['div'], class_='w3-code')
                    if code_box:
                        extracted_parts.append(f"Example:\n{code_box.get_text(strip=True)}")
                elif element.name == 'h2':
                    extracted_parts.append(f"\n## {element.get_text(strip=True)}")
                elif element.name == 'p':
                    extracted_parts.append(element.get_text(strip=True))
                elif element.name == 'pre':
                    extracted_parts.append(f"Code:\n{element.get_text(strip=True)}")

            content = "\n\n".join(filter(None, extracted_parts))
            
            # Form topic list
            topics = [topic] if topic else []
            # topics.extend(self._extract_additional_topics(content))
        
            return {
                "topic": list(set(topics)), # Return distinct list
                "content": content,
                "source": "W3Schools"
            }

        except Exception as e:
            return {"topic": [], "content": "", "source": "W3Schools", "error": str(e)}


class GeeksForGeeksScraper(BaseScraper):
    """
    Scraper for GeeksForGeeks.
    Scrapes concept explanations, short theory, basic algorithms.
    Ignores full code dumps, interview Q&A.
    """
    def fetch(self, query: str) -> dict:
        try:
            url = self._resolve_url(query, site_filter="geeksforgeeks.org")
            soup = self._get_soup(url)
            
            # GFG content structure often changes
            article = soup.find('article') or soup.find('div', class_='text') or soup.find('div', class_='article_content')
            
            if not article:
                return {"topic": "", "content": "", "source": "GeeksForGeeks", "error": "Could not find article content"}

            # Extract Topic
            # Try to find H1 inside article or globally if not found
            h1 = article.find('h1') or soup.find('h1')
            topic = h1.get_text(strip=True) if h1 else ""
            # Don't decompose globally found h1 if it's outside article, but if inside, remove.
            if h1 and h1 in article.descendants:
                h1.decompose()

            # Remove unwanted elements
            for unwanted in article.find_all(['div'], class_=['comments', 'improved', 'print-main', 'share-icons']):
                unwanted.decompose()
            
            extracted_parts = []
            
            for element in article.find_all(['h2', 'h3', 'p', 'ul', 'ol', 'pre']):
                text = element.get_text(strip=True)
                if not text:
                    continue
                
                # Heuristic to skip Interview Questions sections
                if "Interview Questions" in text and element.name in ['h2', 'h3']:
                    break 

                if element.name in ['h2', 'h3']:
                    extracted_parts.append(f"\n### {text}")
                elif element.name in ['ul', 'ol']:
                    for li in element.find_all('li'):
                        extracted_parts.append(f"- {li.get_text(strip=True)}")
                elif element.name == 'pre':
                     extracted_parts.append(f"```\n{text}\n```")
                else:
                    extracted_parts.append(text)

            content = "\n\n".join(extracted_parts)
            
            topics = [topic] if topic else []
            topics.extend(self._extract_additional_topics(content))

            return {
                "topic": list(set(topics)),
                "content": content,
                "source": "GeeksForGeeks"
            }

        except Exception as e:
             return {"topic": [], "content": "", "source": "GeeksForGeeks", "error": str(e)}

class NPTELScraper(BaseScraper):
    """
    Scraper for NPTEL.
    Scrapes lecture transcript text, definition paragraphs.
    Handles PDF if encountered.
    """
    def fetch(self, query: str) -> dict:
        try:
            url = self._resolve_url(query, site_filter="nptel.ac.in")
            
            # Check if PDF
            if url.lower().endswith('.pdf'):
                return self._fetch_pdf(url, "NPTEL")
            
            soup = self._get_soup(url)
            
            content_div = soup.find('div', class_='content') or soup.find('div', id='content') or soup.body

            # Attempt to find topic
            h1 = soup.find('h1') or content_div.find('div', class_='header')
            topic = h1.get_text(strip=True) if h1 else "NPTEL Lecture"

            # Remove metadata
            for unwanted in content_div.find_all(['div'], class_=['header', 'footer', 'nav', 'menu', 'instructor']):
                unwanted.decompose()
            if h1 and h1.name: h1.decompose() # if h1 is a tag

            extracted_text = []
            for p in content_div.find_all('p'):
                text = p.get_text(strip=True)
                # Heuristic to ignore course metadata lines
                if "Course:" in text or "Instructor:" in text:
                    continue
                extracted_text.append(text)

            return {
                "topic": set[topic],
                # "topic": list(set([topic] + self._extract_additional_topics(final_content))),
                "content": final_content,
                "source": "NPTEL"
            }
        except Exception as e:
            return {"topic": [], "content": "", "source": "NPTEL", "error": str(e)}

    def _fetch_pdf(self, url: str, source_name: str) -> dict:
        if not pypdf:
            return {"topic": ["Unknown PDF"], "content": "pypdf not installed, cannot scrape PDF", "source": source_name}
        
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            with io.BytesIO(response.content) as f:
                reader = pypdf.PdfReader(f)
                text = []
                # Try to get metadata title
                topic = reader.metadata.get('/Title', source_name + " PDF") if reader.metadata else source_name + " PDF"
                for page in reader.pages:
                    text.append(page.extract_text())
            
            final_content = "\n".join(text)
            print("✨ Processing additional topics...")
            return {
                "topic": set[topic],
                # "topic": list(set([topic] + self._extract_additional_topics(final_content[:5000]))),
                "content": final_content,
                "source": source_name
            }
        except Exception as e:
             return {"topic": [], "content": "", "source": source_name, "error": str(e)}

class MITOCWScraper(BaseScraper):
    """
    Scraper for MIT OCW.
    Scrapes lecture notes, theory sections.
    """
    def fetch(self, query: str) -> dict:
        try:
            url = self._resolve_url(query, site_filter="ocw.mit.edu")
            
            # Check for PDF lecture notes
            if url.lower().endswith('.pdf'):
                 return self._scrape_pdf(url)

            soup = self._get_soup(url)
            
            main_content = soup.find('main') or soup.find('div', id='course-content-section')
            
            if not main_content:
                return {"topic": "", "content": "", "source": "MITOCW", "error": "Main content not found"}

            # Extract Topic
            h1 = main_content.find('h1')
            topic = h1.get_text(strip=True) if h1 else "MIT OCW Lecture"
            if h1: h1.decompose()

            # Remove syllabus/schedule
            for section in main_content.find_all(['div', 'section']):
                if section.get('id') in ['syllabus', 'calendar', 'schedule']:
                    section.decompose()
            
            # Extract text
            content = []
            for element in main_content.find_all(['h2', 'h3', 'p', 'div']):
                 if element.name == 'div' and not element.get_text(strip=True):
                     continue
                 content.append(element.get_text(strip=True))

            final_content = "\n\n".join(content)
            
            return {
                "topic": set([topic]),
                # "topic": list(set([topic] + self._extract_additional_topics(final_content))),
                "content": final_content,
                "source": "MITOCW"
            }
        except Exception as e:
             return {"topic": [], "content": "", "source": "MITOCW", "error": str(e)}

    def _scrape_pdf(self, url):
        if not pypdf:
            return {"topic": [], "content": "pypdf missing", "source": "MITOCW"}
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            reader = pypdf.PdfReader(io.BytesIO(r.content))
            topic = reader.metadata.get('/Title', "MIT OCW PDF") if reader.metadata else "MIT OCW PDF"
            text = [p.extract_text() for p in reader.pages]
            final_content = "\n".join(text)
            return {"topic": set([topic]), "content": final_content, "source": "MITOCW"}
        except Exception as e:
            return {"topic": [], "content": "", "source": "MITOCW", "error": str(e)}


class OpenStaxScraper(BaseScraper):
    """
    Scraper for OpenStax.
    """
    def fetch(self, query: str) -> dict:
        try:
            url = self._resolve_url(query, site_filter="openstax.org")
            soup = self._get_soup(url)
            
            main_content = soup.find('div', {'data-type': 'page'}) or soup.find('main')

            if not main_content:
                 return {"topic": "", "content": "", "source": "OpenStax", "error": "Content not found"}

            # Topic extraction
            # OpenStax pages often have a title span or h1
            title_elem = main_content.find('span', {'data-type': 'title'}) or main_content.find('h1')
            topic = title_elem.get_text(strip=True) if title_elem else "OpenStax Page"
            # Don't decompose if it's the only reference, or do if inside content.

            extracted = []
            
            # Prioritize definitions
            definitions = main_content.find_all('div', {'data-type': 'definition'})
            for de in definitions:
                extracted.append(f"Definition: {de.get_text(strip=True)}")
            
            intro = main_content.find('div', {'data-type': 'introduction'})
            if intro:
                extracted.append(f"Introduction: {intro.get_text(strip=True)}")
                
            summary = main_content.find('div', {'class': 'summary'})
            if summary:
                 extracted.append(f"Summary: {summary.get_text(strip=True)}")
                 
            if not extracted:
                 count = 0
                 for p in main_content.find_all('p'):
                     extracted.append(p.get_text(strip=True))
                     count += 1
                     if count > 10: 
                         extracted.append("[...Content Trucated...]")
                         break
            
            final_content = "\n\n".join(extracted)
            return {
                "topic": set([topic]),
                # "topic": list(set([topic] + self._extract_additional_topics(final_content))),
                "content": final_content,
                "source": "OpenStax"
            }
        except Exception as e:
            return {"topic": [], "content": "", "source": "OpenStax", "error": str(e)}


class UniversityEDUScraper(BaseScraper):
    """
    Scraper for .edu sites.
    """
    def fetch(self, query: str) -> dict:
        try:
            url = self._resolve_url(query, site_filter=".edu")
            
            if url.lower().endswith('.pdf'):
                return self._scrape_pdf(url)

            soup = self._get_soup(url)
            
            # Topic: Title of the page
            topic = soup.title.get_text(strip=True) if soup.title else "University Page"
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return {
                "topic": set([topic]),
                # "topic": list(set([topic] + self._extract_additional_topics(clean_text))),
                "content": clean_text,
                "source": "UniversityEDU"
            }
            
        except Exception as e:
            return {"topic": [], "content": "", "source": "UniversityEDU", "error": str(e)}

    def _scrape_pdf(self, url):
        if not pypdf:
             return {"topic": [], "content": "pypdf missing", "source": "UniversityEDU"}
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            reader = pypdf.PdfReader(io.BytesIO(r.content))
            topic = reader.metadata.get('/Title', "University PDF") if reader.metadata else "University PDF"
            text = [p.extract_text() for p in reader.pages]
            final_content = "\n".join(text)
            return {"topic":set([topic]), "content": final_content, "source": "UniversityEDU"}
        except Exception as e:
            return {"topic": [], "content": "", "source": "UniversityEDU", "error": str(e)}
