"""European Central Bank speech and monetary policy accounts scraper.

Collects speeches, press conferences, and monetary policy accounts from
the ECB website.  Follows the same pattern as the BoE and Fed scrapers
with rate limiting and exponential backoff retries.

References:
    ECB Speeches: https://www.ecb.europa.eu/press/key/html/index.en.html
    ECB Accounts: https://www.ecb.europa.eu/press/accounts/html/index.en.html
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


class ECBScraper:
    """Scraper for ECB speeches and monetary policy accounts.

    Navigates the ECB website to collect speeches and policy accounts,
    extracting full text content and metadata (date, speaker, title).

    Attributes:
        speeches_url: Base URL for the ECB speeches listing page.
        minutes_url: Base URL for the ECB monetary policy accounts page.
        rate_limit: Minimum seconds between consecutive HTTP requests.
        max_pages: Maximum number of listing pages to scrape.
        timeout: HTTP request timeout in seconds.
        output_dir: Directory to save scraped documents.
    """

    _BASE_URL = "https://www.ecb.europa.eu"
    _USER_AGENT = (
        "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; "
        "+https://warwick.ac.uk)"
    )

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/raw") -> None:
        """Initialise the ECB scraper from pipeline configuration.

        Args:
            config: Full pipeline configuration dict.  The
                ``data.sources.ecb`` sub-key is used.
            output_dir: Directory to write scraped documents.
        """
        ecb_cfg = config["data"]["sources"]["ecb"]
        self.speeches_url: str = ecb_cfg["speeches_url"]
        self.minutes_url: str = ecb_cfg["minutes_url"]
        self.rate_limit: float = config["data"].get("rate_limit_seconds", 1.5)
        self.max_pages: int = ecb_cfg.get("max_pages", 50)
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
        """Collect all ECB speeches and accounts within the date range.

        Returns:
            DataFrame with columns: date, title, speaker, source,
            doc_type, text, url.
        """
        logger.info("--- Collecting European Central Bank documents ---")

        speeches = self._collect_speeches()
        accounts = self._collect_accounts()

        all_docs = speeches + accounts
        if not all_docs:
            logger.warning("No ECB documents collected")
            return pd.DataFrame(
                columns=["date", "title", "speaker", "source", "doc_type", "text", "url"]
            )

        df = pd.DataFrame(all_docs)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        logger.info("Collected %d ECB documents (%d speeches, %d accounts)",
                     len(df), len(speeches), len(accounts))
        return df

    def _collect_speeches(self) -> List[Dict[str, str]]:
        """Scrape ECB speech listing pages and fetch full text.

        Returns:
            List of document dictionaries.
        """
        documents: List[Dict[str, str]] = []
        page = 0  # ECB uses zero-indexed pagination

        while page < self.max_pages:
            logger.debug("Fetching ECB speeches page %d", page)
            try:
                listings = self._fetch_listing_page(self.speeches_url, page)
            except Exception as e:
                logger.warning("Failed to fetch ECB speeches page %d: %s", page, e)
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
                            "speaker": listing.get("speaker", ""),
                            "source": "ecb",
                            "doc_type": "speech",
                            "text": text,
                            "url": listing["url"],
                        })
                except Exception as e:
                    logger.warning("Failed to fetch ECB speech: %s", e)

            if reached_start:
                break
            page += 1

        return documents

    def _collect_accounts(self) -> List[Dict[str, str]]:
        """Scrape ECB monetary policy accounts and fetch full text.

        Returns:
            List of document dictionaries.
        """
        documents: List[Dict[str, str]] = []
        page = 0

        while page < self.max_pages:
            logger.debug("Fetching ECB accounts page %d", page)
            try:
                listings = self._fetch_listing_page(self.minutes_url, page)
            except Exception as e:
                logger.warning("Failed to fetch ECB accounts page %d: %s", page, e)
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
                            "title": listing.get("title", "Monetary Policy Account"),
                            "speaker": "Governing Council",
                            "source": "ecb",
                            "doc_type": "account",
                            "text": text,
                            "url": listing["url"],
                        })
                except Exception as e:
                    logger.warning("Failed to fetch ECB account: %s", e)

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
        """Fetch a single ECB listing page and parse document links.

        Args:
            base_url: Base URL for the listing.
            page: Page number (zero-indexed).

        Returns:
            List of dicts with keys: title, url, date_str, speaker.
        """
        self._throttle()

        # ECB pagination format
        url = f"{base_url}?p={page}"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        listings: List[Dict[str, str]] = []

        # ECB uses definition lists or structured divs
        items = soup.select("div.title, dt, div.list-item, article")
        for item in items:
            link_el = item.select_one("a[href]")
            date_el = item.select_one("time, .date, dd.date")
            speaker_el = item.select_one(".author, .speaker, dd.subtitle")

            if link_el is None:
                continue

            href = link_el.get("href", "")
            if href.startswith("/"):
                href = f"{self._BASE_URL}{href}"

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
        """Fetch and extract main text from an ECB document page.

        Args:
            url: Full URL of the document.

        Returns:
            Cleaned text content.
        """
        self._throttle()

        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        for tag in soup.select("nav, script, style, header, footer, .ecb-langSelector"):
            tag.decompose()

        content = soup.select_one(
            "div.section, article.content, div#main-wrapper, "
            "div.ecb-pressRelease, main"
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
        """Parse a date string in common ECB formats.

        Args:
            date_str: Raw date string from the listing page.

        Returns:
            Parsed datetime, or None if parsing fails.
        """
        formats = [
            "%d %B %Y",       # "15 January 2024"
            "%d/%m/%Y",       # "15/01/2024"
            "%Y-%m-%d",       # "2024-01-15"
            "%d %b %Y",       # "15 Jan 2024"
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
        """Remove excess whitespace and boilerplate.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text string.
        """
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        return text.strip()

    def save(self, df: pd.DataFrame, filename: str = "ecb_documents.parquet") -> Path:
        """Save collected documents to disk.

        Args:
            df: DataFrame of collected documents.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info("Saved %d ECB documents to %s", len(df), output_path)
        return output_path

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
