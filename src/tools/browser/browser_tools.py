import webbrowser
import urllib.parse

def open_url(url: str) -> bool:
    """
    Opens a given URL in the default web browser.
    
    Args:
        url (str): The URL to open (e.g., 'https://www.example.com').
        
    Returns:
        bool: True if the browser opened successfully, False otherwise.
    """
    try:
        # webbrowser.open() returns True if a browser was successfully launched
        return webbrowser.open(url)
    except Exception as e:
        print(f"Error opening URL '{url}': {e}")
        return False

def search_browser(query: str, search_engine: str = 'google') -> bool:
    """
    Searches for a given query in the default web browser.
    
    Args:
        query (str): The search term.
        search_engine (str): The search engine to use ('google', 'bing', 'duckduckgo').
        
    Returns:
        bool: True if the browser opened successfully, False otherwise.
    """
    encoded_query = urllib.parse.quote_plus(query)
    
    engines = {
        'google': f"https://www.google.com/search?q={encoded_query}",
        'bing': f"https://www.bing.com/search?q={encoded_query}",
        'duckduckgo': f"https://duckduckgo.com/?q={encoded_query}"
    }
    
    search_url = engines.get(search_engine.lower(), engines['google'])
    
    try:
        return webbrowser.open(search_url)
    except Exception as e:
        print(f"Error searching for '{query}': {e}")
        return False

# Example usage (can be removed later):
if __name__ == "__main__":
    print("Testing open_url...")
    open_url("https://github.com")
    
    print("Testing search_browser...")
    search_browser("How to build an autonomous robot", "duckduckgo")
