"""Bank of England speech and minutes scraper.

Collects monetary policy speeches, MPC minutes, and monetary policy
reports from the Bank of England website. Handles pagination, rate
limiting, and transient network errors with exponential backoff.

References:
    BoE Speeches: https://www.bankofengland.co.uk/news/speeches
    MPC Minutes: https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class BoEScraper:
    """Scraper for Bank of England speeches and monetary policy documents.

    Navigates the BoE website to collect speeches and MPC minutes,
    extracting the full text content and metadata (date, speaker, title).
    Implements polite scraping with rate limiting and retries.

    Attributes:
        speeches_url: Base URL for the BoE speeches listing page.
        minutes_url: Base URL for the MPC minutes listing page.
        rate_limit: Minimum seconds between consecutive HTTP requests.
        max_pages: Maximum number of listing pages to scrape.
        timeout: HTTP request timeout in seconds.
        output_dir: Directory to save scraped documents.
    """

    _USER_AGENT = (
        "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; "
        "+https://warwick.ac.uk)"
    )

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/raw") -> None:
        """Initialise the BoE scraper from pipeline configuration.

        Args:
            config: Full pipeline configuration dict.  The
                ``data.sources.boe`` sub-key is used.
            output_dir: Directory to write scraped documents.
        """
        boe_cfg = config["data"]["sources"]["boe"]
        self.speeches_url: str = boe_cfg["speeches_url"]
        self.minutes_url: str = boe_cfg["minutes_url"]
        self.rate_limit: float = config["data"].get("rate_limit_seconds", 1.5)
        self.max_pages: int = boe_cfg.get("max_pages", 50)
        self.max_retries: int = config["data"].get("max_retries", 3)
        self.timeout: int = config["data"].get("timeout_seconds", 30)

        self.start_date = datetime.strptime(
            config["data"]["date_range"]["start"], "%Y-%m-%d"
        )
        self.end_date = datetime.strptime(
            config["data"]["date_range"]["end"], "%Y-%m-%d"
        )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self._USER_AGENT})
        self._last_request_time: float = 0.0

    def collect(self) -> pd.DataFrame:
        """Collect all BoE speeches and minutes within the configured date range.

        Returns:
            DataFrame with columns: date, title, speaker, source,
            doc_type, text, url.
        """
        logger.info("--- Collecting Bank of England documents ---")

        speeches = self._collect_speeches()
        minutes = self._collect_minutes()

        all_docs = speeches + minutes
        if not all_docs:
            logger.warning("No BoE documents collected")
            return pd.DataFrame(
                columns=["date", "title", "speaker", "source", "doc_type", "text", "url"]
            )

        df = pd.DataFrame(all_docs)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        logger.info("Collected %d BoE documents (%d speeches, %d minutes)",
                     len(df), len(speeches), len(minutes))
        return df

    def _collect_speeches(self) -> List[Dict[str, str]]:
        """Scrape speech listing pages and fetch full text for each speech.

        Returns:
            List of document dictionaries.
        """
        documents: List[Dict[str, str]] = []
        page = 1

        while page <= self.max_pages:
            logger.debug("Fetching BoE speeches page %d", page)
            try:
                listings = self._fetch_listing_page(self.speeches_url, page)
            except Exception as e:
                logger.warning("Failed to fetch speeches page %d: %s", page, e)
                break

            if not listings:
                logger.debug("No more speech listings found at page %d", page)
                break

            reached_start = False
            for listing in listings:
                doc_date = self._parse_date(listing.get("date_str", ""))
                if doc_date is None:
                    continue
                if doc_date < self.start_date:
                    reached_start = True
                    break
                if doc_date > self.end_date:
                    continue

                try:
                    text = self._fetch_document_text(listing["url"])
                    if text and len(text) >= 100:
                        documents.append({
                            "date": doc_date.strftime("%Y-%m-%d"),
                            "title": listing.get("title", ""),
                            "speaker": listing.get("speaker", ""),
                            "source": "boe",
                            "doc_type": "speech",
                            "text": text,
                            "url": listing["url"],
                        })
                except Exception as e:
                    logger.warning("Failed to fetch speech: %s — %s",
                                   listing.get("title", "unknown"), e)

            if reached_start:
                break
            page += 1

        return documents

    def _collect_minutes(self) -> List[Dict[str, str]]:
        """Scrape MPC minutes listing pages and fetch full text.

        Returns:
            List of document dictionaries.
        """
        documents: List[Dict[str, str]] = []
        page = 1

        while page <= self.max_pages:
            logger.debug("Fetching BoE minutes page %d", page)
            try:
                listings = self._fetch_listing_page(self.minutes_url, page)
            except Exception as e:
                logger.warning("Failed to fetch minutes page %d: %s", page, e)
                break

            if not listings:
                break

            reached_start = False
            for listing in listings:
                doc_date = self._parse_date(listing.get("date_str", ""))
                if doc_date is None:
                    continue
                if doc_date < self.start_date:
                    reached_start = True
                    break
                if doc_date > self.end_date:
                    continue

                try:
                    text = self._fetch_document_text(listing["url"])
                    if text and len(text) >= 100:
                        documents.append({
                            "date": doc_date.strftime("%Y-%m-%d"),
                            "title": listing.get("title", ""),
                            "speaker": "MPC",
                            "source": "boe",
                            "doc_type": "minutes",
                            "text": text,
                            "url": listing["url"],
                        })
                except Exception as e:
                    logger.warning("Failed to fetch minutes document: %s", e)

            if reached_start:
                break
            page += 1

        return documents

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _fetch_listing_page(
        self, base_url: str, page: int
    ) -> List[Dict[str, str]]:
        """Fetch a single listing page and parse document links.

        Args:
            base_url: Base URL for the listing (speeches or minutes).
            page: Page number to fetch.

        Returns:
            List of dicts with keys: title, url, date_str, speaker.
        """
        self._throttle()

        url = f"{base_url}?page={page}"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        listings: List[Dict[str, str]] = []

        # BoE uses <div class="release-list"> or similar containers
        articles = soup.select("div.list-item, article.release, li.results-item")
        for article in articles:
            link_el = article.select_one("a[href]")
            date_el = article.select_one("time, .release-date, .date")
            speaker_el = article.select_one(".speaker, .author, .subtitle")

            if link_el is None:
                continue

            href = link_el.get("href", "")
            if href.startswith("/"):
                href = f"https://www.bankofengland.co.uk{href}"

            listings.append({
                "title": link_el.get_text(strip=True),
                "url": href,
                "date_str": date_el.get_text(strip=True) if date_el else "",
                "speaker": speaker_el.get_text(strip=True) if speaker_el else "",
            })

        return listings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _fetch_document_text(self, url: str) -> str:
        """Fetch and extract the main text content from a document page.

        Args:
            url: Full URL of the document page.

        Returns:
            Cleaned text content of the document.
        """
        self._throttle()

        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Remove navigation, scripts, styles
        for tag in soup.select("nav, script, style, header, footer, .related-links"):
            tag.decompose()

        # Try common BoE content containers
        content = soup.select_one(
            "div.page-content, article.content, div.speech-text, "
            "div#content-body, main"
        )
        if content is None:
            content = soup.body

        if content is None:
            return ""

        text = content.get_text(separator="\n", strip=True)
        text = self._clean_text(text)
        return text

    def _throttle(self) -> None:
        """Enforce rate limiting between consecutive requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse a date string in common BoE formats.

        Args:
            date_str: Raw date string from the listing page.

        Returns:
            Parsed datetime, or None if parsing fails.
        """
        formats = [
            "%d %B %Y",       # "15 January 2024"
            "%d %b %Y",       # "15 Jan 2024"
            "%Y-%m-%d",       # "2024-01-15"
            "%B %Y",          # "January 2024"
        ]
        date_str = date_str.strip()
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove excess whitespace and boilerplate artefacts.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text string.
        """
        # Collapse multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        return text.strip()

    def save(self, df: pd.DataFrame, filename: str = "boe_documents.parquet") -> Path:
        """Save collected documents to disk.

        Args:
            df: DataFrame of collected documents.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info("Saved %d BoE documents to %s", len(df), output_path)
        return output_path

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
