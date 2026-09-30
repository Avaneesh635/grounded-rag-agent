"""Unit tests for crawler and content cleaning."""

import pytest
from src.crawler.cleaner import ContentCleaner, CleanedPage
from src.crawler.crawler import WebCrawler


def test_url_normalization():
    crawler = WebCrawler(
        seed_url="https://fastapi.tiangolo.com/tutorial/",
        allowed_domain="fastapi.tiangolo.com",
        url_path_prefix="/tutorial/",
    )

    # Relative resolution
    norm = crawler.normalize_url(
        "first-steps/", "https://fastapi.tiangolo.com/tutorial/"
    )
    assert norm == "https://fastapi.tiangolo.com/tutorial/first-steps/"

    # Fragment removal
    norm_frag = crawler.normalize_url(
        "path-params/#types", "https://fastapi.tiangolo.com/tutorial/"
    )
    assert norm_frag == "https://fastapi.tiangolo.com/tutorial/path-params/"

    # External domain rejection
    norm_ext = crawler.normalize_url(
        "https://google.com/search", "https://fastapi.tiangolo.com/tutorial/"
    )
    assert norm_ext is None

    # Path prefix rejection
    norm_outside = crawler.normalize_url(
        "https://fastapi.tiangolo.com/benchmarks/",
        "https://fastapi.tiangolo.com/tutorial/",
    )
    assert norm_outside is None

    # Non-HTML extension rejection
    norm_img = crawler.normalize_url(
        "diagram.png", "https://fastapi.tiangolo.com/tutorial/"
    )
    assert norm_img is None


def test_content_cleaner():
    html_sample = """
    <!DOCTYPE html>
    <html>
    <head><title>First Steps - FastAPI</title></head>
    <body>
        <nav class="md-header">Header Navigation</nav>
        <div class="sidebar">Sidebar Links</div>
        <main>
            <h1>First Steps¶</h1>
            <p>The simplest FastAPI file could look like this:</p>
            <code>@app.get("/")</code>
            <p>More detailed text explaining endpoint mechanics and asynchronous functions.</p>
        </main>
        <footer>Footer copyright info</footer>
    </body>
    </html>
    """
    cleaned = ContentCleaner.clean_html(
        html_sample, "https://fastapi.tiangolo.com/tutorial/first-steps/"
    )
    assert cleaned is not None
    assert cleaned.title == "First Steps"
    assert "Header Navigation" not in cleaned.text
    assert "Sidebar Links" not in cleaned.text
    assert "Footer copyright" not in cleaned.text
    assert "simplest FastAPI file" in cleaned.text
    assert cleaned.word_count > 5
