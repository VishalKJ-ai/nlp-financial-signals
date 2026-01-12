"""Federal Reserve speech and FOMC minutes scraper.

Collects monetary policy speeches, FOMC minutes, and statements from
the Federal Reserve website. Implements the same scraping pattern as
the BoE scraper with rate limiting and exponential backoff retries.

References:
    Fed Speeches: https://www.federalreserve.gov/newsevents/speeches.htm
    FOMC Minutes: https://www.federalreserve.gov/monetarypolicy/fomcminutes.htm
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


class FedScraper:
    """Scraper for Federal Reserve speeches and FOMC documents.

    Navigates the Fed website to collect speeches and FOMC minutes,
    extracting the full text content and metadata (date, speaker, title).

    Attributes:
        speeches_url: Base URL for the Fed speeches listing page.
        minutes_url: Base URL for the FOMC minutes listing page.
        rate_limit: Minimum seconds between consecutive HTTP requests.
        max_pages: Maximum number of listing pages to scrape.
        timeout: HTTP request timeout in seconds.
        output_dir: Directory to save scraped documents.
    """

    _BASE_URL = "https://www.federalreserve.gov"
    _USER_AGENT = (
        "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; "
        "+https://warwick.ac.uk)"
    )

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/raw") -> None:
        """Initialise the Fed scraper from pipeline configuration.

        Args:
            config: Full pipeline configuration dict.  The
                ``data.sources.fed`` sub-key is used.
            output_dir: Directory to write scraped documents.
        """
        fed_cfg = config["data"]["sources"]["fed"]
        self.speeches_url: str = fed_cfg["speeches_url"]
        self.minutes_url: str = fed_cfg["minutes_url"]
        self.rate_limit: float = config["data"].get("rate_limit_seconds", 1.5)
        self.max_pages: int = fed_cfg.get("max_pages", 50)
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
        """Collect all Fed speeches and FOMC minutes within the date range.

        Returns:
            DataFrame with columns: date, title, speaker, source,
            doc_type, text, url.
        """
        logger.info("--- Collecting Federal Reserve documents ---")

        speeches = self._collect_speeches()
        minutes = self._collect_minutes()

        all_docs = speeches + minutes
        if not all_docs:
            logger.warning("No Fed documents collected")
            return pd.DataFrame(
                columns=["date", "title", "speaker", "source", "doc_type", "text", "url"]
            )

        df = pd.DataFrame(all_docs)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        logger.info("Collected %d Fed documents (%d speeches, %d minutes)",
                     len(df), len(speeches), len(minutes))
        return df

    def _collect_speeches(self) -> List[Dict[str, str]]:
        """Scrape Fed speech listing pages and fetch full text.

        Returns:
            List of document dictionaries.
        """
        documents: List[Dict[str, str]] = []

        # The Fed speeches page uses year-based filtering
        for year in range(self.end_date.year, self.start_date.year - 1, -1):
            logger.debug("Fetching Fed speeches for year %d", year)
            try:
                listings = self._fetch_speech_listing(year)
            except Exception as e:
                logger.warning("Failed to fetch Fed speeches for %d: %s", year, e)
                continue

            for listing in listings:
                doc_date = self._parse_date(listing.get("date_str", ""))
                if doc_date is None:
                    continue
                if doc_date < self.start_date or doc_date > self.end_date:
                    continue

                try:
                    text = self._fetch_document_text(listing["url"])
                    if text and len(text) >= 100:
                        documents.append({
                            "date": doc_date.strftime("%Y-%m-%d"),
                            "title": listing.get("title", ""),
                            "speaker": listing.get("speaker", ""),
                            "source": "fed",
                            "doc_type": "speech",
                            "text": text,
                            "url": listing["url"],
                        })
                except Exception as e:
                    logger.warning("Failed to fetch Fed speech: %s", e)

        return documents

    def _collect_minutes(self) -> List[Dict[str, str]]:
        """Scrape FOMC minutes listing pages and fetch full text.

        Returns:
            List of document dictionaries.
        """
        documents: List[Dict[str, str]] = []

        for year in range(self.end_date.year, self.start_date.year - 1, -1):
            logger.debug("Fetching FOMC minutes for year %d", year)
            try:
                listings = self._fetch_minutes_listing(year)
            except Exception as e:
                logger.warning("Failed to fetch FOMC minutes for %d: %s", year, e)
                continue

            for listing in listings:
                doc_date = self._parse_date(listing.get("date_str", ""))
                if doc_date is None:
                    continue
                if doc_date < self.start_date or doc_date > self.end_date:
                    continue

                try:
                    text = self._fetch_document_text(listing["url"])
                    if text and len(text) >= 100:
                        documents.append({
                            "date": doc_date.strftime("%Y-%m-%d"),
                            "title": listing.get("title", "FOMC Minutes"),
                            "speaker": "FOMC",
                            "source": "fed",
                            "doc_type": "minutes",
                            "text": text,
                            "url": listing["url"],
                        })
                except Exception as e:
                    logger.warning("Failed to fetch FOMC minutes: %s", e)

        return documents

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _fetch_speech_listing(self, year: int) -> List[Dict[str, str]]:
        """Fetch the speech listing page for a given year.

        Args:
            year: Calendar year to fetch.

        Returns:
            List of dicts with keys: title, url, date_str, speaker.
        """
        self._throttle()

        url = f"{self.speeches_url}/{year}-speeches.htm"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        listings: List[Dict[str, str]] = []

        rows = soup.select("div.row.eventlist, tr, div.fomc-meeting")
        for row in rows:
            link_el = row.select_one("a[href]")
            date_el = row.select_one("time, .eventlist__date, td:first-child")
            speaker_el = row.select_one(".speaker, .news__speaker")

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
    def _fetch_minutes_listing(self, year: int) -> List[Dict[str, str]]:
        """Fetch the FOMC minutes listing page for a given year.

        Args:
            year: Calendar year to fetch.

        Returns:
            List of dicts with keys: title, url, date_str.
        """
        self._throttle()

        url = f"{self.minutes_url}/{year}.htm"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        listings: List[Dict[str, str]] = []

        links = soup.select("a[href*='minutes']")
        for link in links:
            href = link.get("href", "")
            if href.startswith("/"):
                href = f"{self._BASE_URL}{href}"

            # Try to find sibling or parent date element
            parent = link.find_parent(["tr", "div", "li"])
            date_str = ""
            if parent:
                date_el = parent.select_one("time, td:first-child, .date")
                date_str = date_el.get_text(strip=True) if date_el else ""

            listings.append({
                "title": link.get_text(strip=True) or "FOMC Minutes",
                "url": href,
                "date_str": date_str,
            })

        return listings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _fetch_document_text(self, url: str) -> str:
        """Fetch and extract the main text from a Fed document page.

        Args:
            url: Full URL of the document.

        Returns:
            Cleaned text content.
        """
        self._throttle()

        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        for tag in soup.select("nav, script, style, header, footer"):
            tag.decompose()

        content = soup.select_one(
            "div.col-xs-12.col-sm-8.col-md-9, div#article, "
            "div.fomc-minutes, div#content"
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
        """Parse a date string in common Fed formats.

        Args:
            date_str: Raw date string from the listing page.

        Returns:
            Parsed datetime, or None if parsing fails.
        """
        formats = [
            "%B %d, %Y",      # "January 15, 2024"
            "%b %d, %Y",      # "Jan 15, 2024"
            "%Y-%m-%d",       # "2024-01-15"
            "%m/%d/%Y",       # "01/15/2024"
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

    def save(self, df: pd.DataFrame, filename: str = "fed_documents.parquet") -> Path:
        """Save collected documents to disk.

        Args:
            df: DataFrame of collected documents.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info("Saved %d Fed documents to %s", len(df), output_path)
        return output_path

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
