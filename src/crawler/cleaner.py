"""HTML content extraction and cleaning for website pages."""

import re
from dataclasses import dataclass
from typing import Optional
from bs4 import BeautifulSoup
import trafilatura


@dataclass
class CleanedPage:
    """Represents cleaned and extracted content from a webpage."""

    url: str
    title: str
    description: str
    text: str
    char_count: int
    word_count: int


class ContentCleaner:
    """Extracts high-signal textual content while removing boilerplate and navigation."""

    @staticmethod
    def extract_title(soup: BeautifulSoup, url: str) -> str:
        """Extract a readable title for the page."""
        # 1. Check <title> tag
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            raw_title = title_tag.get_text(strip=True)
            # Remove common suffixes like " - FastAPI"
            clean_title = re.split(r"[-–—|]", raw_title)[0].strip()
            if clean_title:
                return clean_title

        # 2. Check <h1> tag
        h1_tag = soup.find("h1")
        if h1_tag and h1_tag.get_text(strip=True):
            return h1_tag.get_text(strip=True).rstrip("¶").strip()

        # 3. Fallback to path segment
        path_slug = url.rstrip("/").split("/")[-1].replace("-", " ").title()
        return path_slug or "Documentation Page"

    @staticmethod
    def extract_description(soup: BeautifulSoup) -> str:
        """Extract meta description if available."""
        desc_meta = soup.find("meta", attrs={"name": "description"})
        if desc_meta and desc_meta.get("content"):
            return desc_meta["content"].strip()
        return ""

    @classmethod
    def clean_html(cls, html: str, url: str) -> Optional[CleanedPage]:
        """Extract and clean text from raw HTML."""
        if not html or not html.strip():
            return None

        soup = BeautifulSoup(html, "html.parser")
        title = cls.extract_title(soup, url)
        description = cls.extract_description(soup)

        # Primary extraction: trafilatura (best-in-class for article/doc body)
        extracted_text = trafilatura.extract(
            html,
            include_links=False,
            include_images=False,
            favor_precision=True,
            output_format="txt",
        )

        # Fallback to BeautifulSoup if trafilatura extracts too little
        if not extracted_text or len(extracted_text.strip()) < 120:
            # Strip unwanted elements
            for tag in soup(
                ["script", "style", "nav", "header", "footer", "aside", "form"]
            ):
                tag.decompose()
            for cls_to_remove in [
                "md-header",
                "md-sidebar",
                "md-footer",
                "navbar",
                "sidebar",
            ]:
                for el in soup.find_all(class_=cls_to_remove):
                    el.decompose()
            extracted_text = soup.get_text(separator="\n", strip=True)

        if not extracted_text:
            return None

        # Clean up text artifacts
        # 1. Remove dangling anchor symbols (MkDocs/Sphinx '¶')
        text = re.sub(r"¶", "", extracted_text)
        # 2. Collapse excessive whitespace and blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        if len(text) < 50:
            return None

        words = text.split()

        return CleanedPage(
            url=url,
            title=title,
            description=description,
            text=text,
            char_count=len(text),
            word_count=len(words),
        )
