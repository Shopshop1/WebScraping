import requests
from bs4 import BeautifulSoup


class WebScraping:

    def __init__(self, url):
        self.url = url

    def url2paragraphs(self):
        # Send a GET request to the website
        response = requests.get(self.url)

        # Create a BeautifulSoup object to parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all the paragraphs, headings, and list items
        elements = soup.find_all(['p', 'ul'])

        # Extract the text content and their respective tags
        content = []
        for element in elements:
            tag = element.name
            if tag == 'ul':
                # Preserve line breaks within <ul> elements
                text = '\n'.join([li.get_text() for li in element.find_all('li')])  # join list items with a newline
            else:
                text = element.get_text()
            if len(text.strip()) > 0:  # remove empty strings
                content.append(text.strip())

        return content

