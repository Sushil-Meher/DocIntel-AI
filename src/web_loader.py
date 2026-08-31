import requests
from bs4 import BeautifulSoup

from .document_loader import Document


def load_webpage(url: str) -> Document:
    response = requests.get(
        url,
        timeout=15,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    text = soup.get_text(
        separator=" ",
        strip=True
    )

    return Document(
        text=text,
        source=url,
        page=1
    )


if __name__ == "__main__":

    url = "https://example.com"

    document = load_webpage(url)

    print("SOURCE:")
    print(document["source"])

    print("\nTEXT:")
    print(document["text"][:2000])