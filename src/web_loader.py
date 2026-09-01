import requests
from bs4 import BeautifulSoup

from .document_loader import Document


# Below this many words, treat the page as unusable rather than silently
# building an index out of an empty/near-empty document (chunk_document
# would just return no chunks, and create_index([]) crashes on an empty
# embedding array with a cryptic numpy error).
MIN_WORDS = 20


def load_webpage(url: str) -> Document:
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

    except requests.exceptions.MissingSchema:
        raise ValueError(
            f"'{url}' is missing http:// or https://. Try https://{url}"
        )

    except requests.exceptions.InvalidURL:
        raise ValueError(f"'{url}' is not a valid URL.")

    except requests.exceptions.ConnectionError:
        raise ValueError(
            f"Could not connect to '{url}'. Check the address and try again."
        )

    except requests.exceptions.Timeout:
        raise ValueError(f"'{url}' took too long to respond.")

    except requests.exceptions.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        raise ValueError(f"'{url}' returned an error (HTTP {status}).")

    except requests.exceptions.RequestException as error:
        raise ValueError(f"Could not load '{url}': {error}")

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        element.decompose()

    text = soup.get_text(
        separator=" ",
        strip=True
    )

    if len(text.split()) < MIN_WORDS:
        raise ValueError(
            f"'{url}' doesn't have enough readable text to answer "
            "questions about (it may be blank or mostly navigation/scripts)."
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
    print(document.source)

    print("\nTEXT:")
    print(document.text[:2000])