"""Polite, boundary-respecting web crawler for technical documentation websites."""

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console

from src.config import settings
from src.crawler.cleaner import CleanedPage, ContentCleaner

logger = logging.getLogger(__name__)
console = Console()


class WebCrawler:
    """BFS Web Crawler that extracts content within domain and path boundaries."""

    def __init__(
        self,
        seed_url: Optional[str] = None,
        allowed_domain: Optional[str] = None,
        url_path_prefix: Optional[str] = None,
        max_pages: Optional[int] = None,
        delay_seconds: Optional[float] = None,
    ):
        if seed_url:
            self.seed_url = seed_url
            parsed = urlparse(seed_url)
            self.allowed_domain = allowed_domain or parsed.netloc
            # Default to path prefix if meaningful, otherwise empty string for whole domain
            if url_path_prefix is not None:
                self.url_path_prefix = url_path_prefix
            else:
                self.url_path_prefix = (
                    parsed.path if parsed.path and parsed.path != "/" else ""
                )
        else:
            self.seed_url = settings.SEED_URL
            self.allowed_domain = allowed_domain or settings.ALLOWED_DOMAIN
            self.url_path_prefix = (
                url_path_prefix
                if url_path_prefix is not None
                else settings.URL_PATH_PREFIX
            )

        self.max_pages = max_pages or settings.MAX_PAGES
        self.delay_seconds = (
            delay_seconds if delay_seconds is not None else settings.CRAWL_DELAY_SECONDS
        )

        self.visited_urls: Set[str] = set()
        self.crawled_pages: List[CleanedPage] = []
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": settings.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )

    def normalize_url(self, raw_url: str, base_url: str) -> Optional[str]:
        """Normalize URL, removing fragments and tracking params while resolving relative paths."""
        try:
            # Resolve relative URLs
            absolute_url = urljoin(base_url, raw_url)
            # Remove fragment/hash
            defragged, _ = urldefrag(absolute_url)
            parsed = urlparse(defragged)

            # Enforce http or https
            if parsed.scheme not in ("http", "https"):
                return None

            # Enforce domain
            netloc = parsed.netloc.lower()
            if (
                self.allowed_domain
                and netloc != self.allowed_domain
                and not netloc.endswith("." + self.allowed_domain)
            ):
                return None

            # Enforce path prefix if configured
            path = parsed.path
            if self.url_path_prefix and not path.startswith(self.url_path_prefix):
                return None

            # Skip common non-html extensions
            skip_extensions = (
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".svg",
                ".ico",
                ".pdf",
                ".zip",
                ".tar",
                ".gz",
                ".json",
                ".xml",
                ".mp4",
                ".mp3",
                ".css",
                ".js",
            )
            if any(path.lower().endswith(ext) for ext in skip_extensions):
                return None

            # Clean path (ensure trailing slash for directory style doc URLs)
            if not path.endswith("/") and not os.path.splitext(path)[1]:
                path += "/"

            normalized = f"{parsed.scheme}://{netloc}{path}"
            return normalized
        except Exception:
            return None

    def extract_internal_links(self, html: str, current_url: str) -> List[str]:
        """Extract valid internal links from HTML."""
        links: List[str] = []
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                normalized = self.normalize_url(href, current_url)
                if normalized and normalized not in self.visited_urls:
                    links.append(normalized)
        except Exception as e:
            logger.warning("Error parsing links from %s: %s", current_url, e)
        return links

    def crawl_page(self, url: str) -> Optional[tuple[CleanedPage, List[str]]]:
        """Fetch a single page, clean its content, and extract internal links."""
        try:
            time.sleep(self.delay_seconds)
            response = self.session.get(
                url, timeout=settings.CRAWL_TIMEOUT_SECONDS, allow_redirects=True
            )

            if response.status_code != 200:
                logger.warning("HTTP %d for URL: %s", response.status_code, url)
                return None

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                logger.debug("Skipping non-HTML content (%s) for %s", content_type, url)
                return None

            html = response.text
            final_url = response.url

            # Clean and extract content
            cleaned_page = ContentCleaner.clean_html(html, final_url)
            if not cleaned_page:
                logger.debug("Insufficient text content extracted from %s", final_url)
                return None

            # Discover links
            internal_links = self.extract_internal_links(html, final_url)
            return cleaned_page, internal_links

        except requests.RequestException as e:
            logger.error("Failed to fetch %s: %s", url, e)
            return None

    def crawl(self, force_recrawl: bool = False) -> List[CleanedPage]:
        """Run BFS crawler up to max_pages."""
        # Check cache if available and not forced
        if not force_recrawl and settings.CRAWLED_DATA_FILE.exists():
            console.print(
                f"[bold green]Loading cached crawled data from {settings.CRAWLED_DATA_FILE}...[/bold green]"
            )
            cached = self.load_crawled_data()
            if cached and len(cached) >= min(15, self.max_pages):
                console.print(f"[green]Loaded {len(cached)} pages from cache.[/green]")
                self.crawled_pages = cached
                return self.crawled_pages

        console.print(
            f"[bold cyan]Starting BFS Crawl on {self.seed_url} (Max: {self.max_pages} pages)...[/bold cyan]"
        )
        queue: deque[str] = deque([self.seed_url])
        self.visited_urls.clear()
        self.crawled_pages.clear()

        while queue and len(self.crawled_pages) < self.max_pages:
            current_url = queue.popleft()
            if current_url in self.visited_urls:
                continue

            self.visited_urls.add(current_url)
            console.print(
                f"[{len(self.crawled_pages) + 1}/{self.max_pages}] Crawling: [blue]{current_url}[/blue]"
            )

            result = self.crawl_page(current_url)
            if not result:
                continue

            cleaned_page, new_links = result
            self.crawled_pages.append(cleaned_page)
            console.print(
                f"  ✓ Extracted: [green]{cleaned_page.title}[/green] ({cleaned_page.word_count} words)"
            )

            for link in new_links:
                if link not in self.visited_urls and link not in queue:
                    queue.append(link)

        console.print(
            f"\n[bold green]Crawl completed! Successfully indexed {len(self.crawled_pages)} pages.[/bold green]"
        )
        self.save_crawled_data()
        return self.crawled_pages

    def save_crawled_data(self, output_path: Optional[Path] = None) -> None:
        """Save crawled pages to JSON for caching and reproducibility."""
        path = output_path or settings.CRAWLED_DATA_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "url": p.url,
                "title": p.title,
                "description": p.description,
                "text": p.text,
                "char_count": p.char_count,
                "word_count": p.word_count,
            }
            for p in self.crawled_pages
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        console.print(f"Saved crawled data to [cyan]{path}[/cyan]")

    def load_crawled_data(self, input_path: Optional[Path] = None) -> List[CleanedPage]:
        """Load cached crawled pages from JSON."""
        path = input_path or settings.CRAWLED_DATA_FILE
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pages = [
            CleanedPage(
                url=item["url"],
                title=item["title"],
                description=item.get("description", ""),
                text=item["text"],
                char_count=item["char_count"],
                word_count=item["word_count"],
            )
            for item in data
        ]
        return pages
