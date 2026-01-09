
import unittest
from unittest.mock import patch, MagicMock
from scrapers import (
    W3SchoolsScraper,
    GeeksForGeeksScraper,
    NPTELScraper,
    MITOCWScraper,
    OpenStaxScraper,
    UniversityEDUScraper
)
import io
import sys

class TestScrapers(unittest.TestCase):

    def setUp(self):
        self.w3_scraper = W3SchoolsScraper()
        self.gfg_scraper = GeeksForGeeksScraper()
        self.nptel_scraper = NPTELScraper()
        self.mit_scraper = MITOCWScraper()
        self.stax_scraper = OpenStaxScraper()
        self.edu_scraper = UniversityEDUScraper()

    def _print_result(self, scraper_name, url, result):
        print(f"\n{'-'*80}", flush=True)
        print(f"Testing Scraper: {scraper_name}", flush=True)
        print(f"URL: {url}", flush=True)
        print(f"{'-'*80}", flush=True)
        print(f"TOPIC:   {result.get('topic', 'N/A')}", flush=True)
        print(f"SOURCE:  {result.get('source', 'N/A')}", flush=True)
        print(f"CONTENT:\n{result.get('content', '')}", flush=True)
        print(f"{'-'*80}\n", flush=True)

    @patch('requests.get')
    def test_w3schools_scraper(self, mock_get):
        url = "https://www.w3schools.com/python/python_lists.asp"
        html_content = """
        <html>
        <div id="main">
            <h1>Python Lists</h1>
            <div class="w3-col">Ad</div>
            <p>Lists are used to store multiple items.</p>
            <div class="w3-example">
                <div class="w3-code">mylist = ["apple", "banana", "cherry"]</div>
            </div>
        </div>
        </html>
        """
        mock_response = MagicMock()
        mock_response.content = html_content.encode('utf-8')
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = self.w3_scraper.fetch(url)
        self._print_result("W3Schools", url, result)
        
        self.assertEqual(result['source'], "W3Schools")
        self.assertEqual(result['topic'], "Python Lists")
        self.assertIn("Lists are used to store multiple items.", result['content'])
        self.assertIn('mylist = ["apple", "banana", "cherry"]', result['content'])

    @patch('requests.get')
    def test_gfg_scraper(self, mock_get):
        url = "https://www.geeksforgeeks.org/binary-search/"
        html_content = """
        <html>
        <article>
            <h1>Binary Search</h1>
            <div class="comments">User comments</div>
            <p>Binary Search is a sorting algorithm.</p>
            <pre>def binary_search(arr, x): ...</pre>
        </article>
        </html>
        """
        mock_response = MagicMock()
        mock_response.content = html_content.encode('utf-8')
        mock_get.return_value = mock_response

        result = self.gfg_scraper.fetch(url)
        self._print_result("GeeksForGeeks", url, result)
        
        self.assertEqual(result['source'], "GeeksForGeeks")
        self.assertEqual(result['topic'], "Binary Search")
        self.assertIn("sorting algorithm", result['content'])
        self.assertNotIn("User comments", result['content'])

    @patch('requests.get')
    def test_nptel_scraper_html(self, mock_get):
        url = "https://nptel.ac.in/course/123"
        html_content = """
        <html>
        <div class="content">
            <div class="header">Header</div>
            <h1>Course Introduction</h1>
            <p>Welcome to the course.</p>
            <p>Lecture 1: Intro.</p>
            <div class="footer">Footer</div>
        </div>
        </html>
        """
        mock_response = MagicMock()
        mock_response.content = html_content.encode('utf-8')
        mock_get.return_value = mock_response
        
        result = self.nptel_scraper.fetch(url)
        self._print_result("NPTEL", url, result)

        self.assertEqual(result['source'], "NPTEL")
        self.assertEqual(result['topic'], "Course Introduction")
        self.assertIn("Welcome to the course.", result['content'])
        self.assertNotIn("Header", result['content'])

    @patch('requests.get')
    def test_mitocw_scraper(self, mock_get):
        url = "https://ocw.mit.edu/courses/18-06/1"
        html_content = """
        <html>
        <main>
            <div id="syllabus">Syllabus content</div>
            <h1>Lecture 1: LA</h1>
            <p>Linear Algebra basics.</p>
        </main>
        </html>
        """
        mock_response = MagicMock()
        mock_response.content = html_content.encode('utf-8')
        mock_get.return_value = mock_response
        
        result = self.mit_scraper.fetch(url)
        self._print_result("MITOCW", url, result)

        self.assertEqual(result['source'], "MITOCW")
        self.assertEqual(result['topic'], "Lecture 1: LA")
        self.assertIn("Linear Algebra basics.", result['content'])
        self.assertNotIn("Syllabus content", result['content'])

    @patch('requests.get')
    def test_openstax_scraper(self, mock_get):
        url = "https://openstax.org/books/biology/pages/1-1"
        html_content = """
        <html>
        <div data-type="page">
            <span data-type="title">Ecology Basics</span>
            <div data-type="definition">Def 1: Ecology</div>
            <div data-type="introduction">Intro text</div>
            <div class="summary">Summary text</div>
            <p>Chapter body...</p>
        </div>
        </html>
        """
        mock_response = MagicMock()
        mock_response.content = html_content.encode('utf-8')
        mock_get.return_value = mock_response
        
        result = self.stax_scraper.fetch(url)
        self._print_result("OpenStax", url, result)

        self.assertEqual(result['source'], "OpenStax")
        self.assertEqual(result['topic'], "Ecology Basics")
        self.assertIn("Def 1: Ecology", result['content'])
        self.assertIn("Intro text", result['content'])
        self.assertIn("Summary text", result['content'])

    @patch('requests.get')
    def test_university_edu_scraper(self, mock_get):
        url = "https://cs.stanford.edu/class.html"
        html_content = """
        <html>
        <head><title>CS101 Physics Notes</title></head>
        <body>
            <nav>Menu</nav>
            <p>Lecture notes on Physics.</p>
            <script>var x=1;</script>
        </body>
        </html>
        """
        mock_response = MagicMock()
        mock_response.content = html_content.encode('utf-8')
        mock_get.return_value = mock_response
        
        result = self.edu_scraper.fetch(url)
        self._print_result("UniversityEDU", url, result)

        self.assertEqual(result['source'], "UniversityEDU")
        self.assertEqual(result['topic'], "CS101 Physics Notes") # Title tag
        self.assertIn("Lecture notes on Physics.", result['content'])
        self.assertNotIn("Menu", result['content']) # nav removed
        self.assertNotIn("var x=1", result['content']) # script removed

if __name__ == '__main__':
    # Use TextTestRunner with stdout to ensure synchronization with prints
    suite = unittest.TestLoader().loadTestsFromTestCase(TestScrapers)
    unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
