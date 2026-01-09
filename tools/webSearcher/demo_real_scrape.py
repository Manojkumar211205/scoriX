
import sys
from scrapers import W3SchoolsScraper, GeeksForGeeksScraper

def print_separator(title):
    print(f"\n{'='*80}")
    print(f" REAL SCRAPE DEMO: {title}")
    print(f"{'='*80}")

def run_demo():
    # 1. W3Schools
    w3 = W3SchoolsScraper()
    w3_url = "https://www.w3schools.com/python/python_lists.asp"
    print_separator("W3Schools")
    print(f"Fetching: {w3_url} ...")
    try:
        result = w3.fetch(w3_url)
        print(f"TOPIC: {result.get('topic')}")
        content = result.get('content', '')
        print(f"CONTENT LENGTH: {len(content)} chars")
        print("-" * 40)
        print("FULL CONTENT:")
        print(content)
    except Exception as e:
        print(f"Error: {e}")

    # 2. GeeksForGeeks
    gfg = GeeksForGeeksScraper()
    gfg_url = "https://www.geeksforgeeks.org/binary-search/"
    print_separator("GeeksForGeeks")
    print(f"Fetching: {gfg_url} ...")
    try:
        result = gfg.fetch(gfg_url)
        print(f"TOPIC: {result.get('topic')}")
        content = result.get('content', '')
        print(f"CONTENT LENGTH: {len(content)} chars")
        print("-" * 40)
        print("FULL CONTENT:")
        print(content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_demo()
