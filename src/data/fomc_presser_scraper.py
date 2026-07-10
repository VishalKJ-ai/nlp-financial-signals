"""FOMC press conference transcript scraper.

Collects the full series of FOMC post-meeting press conference
transcripts from the Federal Reserve website, from the first Bernanke
press conference (27 April 2011) to the present.  This is the corpus
specified in Chapter 3 of the dissertation: a single, register-consistent
document type spanning multiple monetary policy cycles.

Discovery uses two listing pages:
    Calendars:  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    Historical: https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm

Transcript PDFs follow a stable naming convention:
    https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf

After download, each PDF is parsed into speaker turns.  Turns by the
Chair are labelled as either the prepared opening statement or a Q&A
answer, so downstream analysis can restrict to spontaneous answers as
required by the methodology.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
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


class FomcPressConfScraper:
    """Scraper for FOMC press conference transcript PDFs.

    Discovers press conference dates from the Fed's calendar and
    historical materials pages, downloads the transcript PDFs with
    rate limiting, and parses them into a speaker-turn corpus.

    Attributes:
        rate_limit: Minimum seconds between consecutive HTTP requests.
        timeout: HTTP request timeout in seconds.
        output_dir: Directory to save downloaded PDFs.
    """

    _BASE_URL = "https://www.federalreserve.gov"
    _CALENDARS_URL = f"{_BASE_URL}/monetarypolicy/fomccalendars.htm"
    _HISTORICAL_URL = f"{_BASE_URL}/monetarypolicy/fomchistorical{{year}}.htm"
    _PDF_URL = f"{_BASE_URL}/mediacenter/files/FOMCpresconf{{date}}.pdf"
    _USER_AGENT = (
        "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; "
        "+https://warwick.ac.uk)"
    )

    # First press conference: Bernanke, 27 April 2011.
    _FIRST_YEAR = 2011

    # Speaker labels sit at the start of a line as an all-caps name
    # terminated by a period, e.g. "CHAIR POWELL." or "STEVE LIESMAN."
    _SPEAKER_RE = re.compile(
        r"^(?P<name>[A-Z][A-Z .’'\-]{2,40}?)\.\s+", re.MULTILINE
    )

    # Page furniture repeated on every transcript page.  The running
    # header ("March 21, 2018 Chairman Powell's Press Conference FINAL
    # Page 5 of 22") is not line-anchored: pypdf splices it into the
    # middle of sentences that span a page break, so it must be removed
    # wherever it appears in the text stream.
    _MONTHS = (r"(?:January|February|March|April|May|June|July|August|"
               r"September|October|November|December)")
    _PAGE_NOISE_RES = [
        re.compile(
            rf"(?:{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\s+)?"
            r"Chair(?:man|woman)?\s+[A-Z][A-Za-z]+[’']?s?\s+"
            r"Press\s*Conference\s*(?:FINAL|PRELIMINARY)?\s*"
            r"(?:Page\s+\d+\s+of\s+\d+)?"
        ),
        re.compile(r"Page \d+ of \d+"),
        re.compile(
            r"^(?:PRELIMINARY|FINAL)?\s*Transcript of (?:Chair|Chairman)"
            r".{0,60}Press Conference.*$",
            re.MULTILINE,
        ),
        re.compile(rf"^{_MONTHS} \d{{1,2}},? \d{{4}}$", re.MULTILINE),
    ]

    def __init__(
        self, config: Dict[str, Any], output_dir: str = "data/raw/fomc_pressconf"
    ) -> None:
        """Initialise the scraper from pipeline configuration.

        Args:
            config: Full pipeline configuration dict.  ``data.rate_limit_seconds``
                and ``data.timeout_seconds`` are used when present.
            output_dir: Directory to save transcript PDFs.
        """
        data_cfg = config.get("data", {})
        self.rate_limit: float = float(data_cfg.get("rate_limit_seconds", 1.5))
        self.timeout: int = int(data_cfg.get("timeout_seconds", 30))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = self._USER_AGENT
        self._last_request: float = 0.0

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep so consecutive requests respect the rate limit."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _get(self, url: str) -> requests.Response:
        """GET a URL with throttling and exponential-backoff retries."""
        self._throttle()
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_dates(self) -> List[str]:
        """Discover all press conference dates as YYYYMMDD strings.

        Scans the calendars page plus every historical year page from
        2011 onward.  Missing historical pages (recent years are only
        on the calendars page) are skipped silently.

        Returns:
            Sorted list of unique YYYYMMDD date strings.
        """
        link_re = re.compile(r"fomcpresconf(\d{8})\.htm", re.IGNORECASE)
        dates: set = set()

        pages = [self._CALENDARS_URL] + [
            self._HISTORICAL_URL.format(year=year)
            for year in range(self._FIRST_YEAR, pd.Timestamp.now().year + 1)
        ]
        for url in pages:
            try:
                html = self._get(url).text
            except requests.RequestException:
                logger.debug("Listing page unavailable (expected for "
                             "recent years): %s", url)
                continue
            found = link_re.findall(html)
            if found:
                logger.info("Found %d press conference links on %s",
                            len(found), url)
            dates.update(found)

        discovered = sorted(dates)
        logger.info("Discovered %d unique press conference dates "
                    "(%s to %s)", len(discovered),
                    discovered[0] if discovered else "n/a",
                    discovered[-1] if discovered else "n/a")
        return discovered

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self, dates: Optional[List[str]] = None) -> List[Path]:
        """Download transcript PDFs for the given dates.

        Already-downloaded files are skipped, making the operation
        resumable.

        Args:
            dates: YYYYMMDD strings; discovered automatically if None.

        Returns:
            Paths of all transcript PDFs present after the run.
        """
        dates = dates if dates is not None else self.discover_dates()
        paths: List[Path] = []
        for date in dates:
            path = self.output_dir / f"FOMCpresconf{date}.pdf"
            if path.exists() and path.stat().st_size > 0:
                paths.append(path)
                continue
            url = self._PDF_URL.format(date=date)
            try:
                response = self._get(url)
            except requests.RequestException as exc:
                logger.warning("Failed to download %s: %s", url, exc)
                continue
            path.write_bytes(response.content)
            logger.info("Downloaded %s (%d KB)", path.name,
                        len(response.content) // 1024)
            paths.append(path)
        logger.info("Corpus contains %d transcript PDFs", len(paths))
        return paths

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        """Extract raw text from a transcript PDF."""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _clean_page_noise(self, text: str) -> str:
        """Remove repeated page headers, footers, and page numbers."""
        for pattern in self._PAGE_NOISE_RES:
            text = pattern.sub("", text)
        return text

    @staticmethod
    def _classify_speaker(name: str) -> str:
        """Classify a speaker label as chair, moderator, or journalist."""
        if name.startswith(("CHAIR", "CHAIRMAN")):
            return "chair"
        if "MICHELLE SMITH" in name:
            return "moderator"
        return "journalist"

    def parse_transcript(self, path: Path) -> pd.DataFrame:
        """Parse one transcript PDF into a speaker-turn DataFrame.

        The Chair's first contiguous block of turns (before any other
        speaker) is labelled ``opening_statement``; every subsequent
        Chair turn is a ``qa_answer``.

        Args:
            path: Path to a FOMCpresconf{date}.pdf file.

        Returns:
            DataFrame with columns: date, speaker, role, segment_type,
            turn_index, text.
        """
        date = re.search(r"(\d{8})", path.name).group(1)
        raw = self._clean_page_noise(self._extract_pdf_text(path))

        matches = list(self._SPEAKER_RE.finditer(raw))
        if not matches:
            logger.warning("No speaker labels found in %s", path.name)
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        qa_started = False
        for i, match in enumerate(matches):
            name = " ".join(match.group("name").split())
            role = self._classify_speaker(name)
            if role != "chair":
                qa_started = True
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            text = " ".join(raw[start:end].split())
            if not text:
                continue
            segment_type = (
                "opening_statement" if role == "chair" and not qa_started
                else "qa_answer" if role == "chair"
                else "question"
            )
            rows.append({
                "date": pd.to_datetime(date, format="%Y%m%d"),
                "speaker": name,
                "role": role,
                "segment_type": segment_type,
                "turn_index": i,
                "text": text,
            })
        return pd.DataFrame(rows)

    def build_corpus(self, paths: Optional[List[Path]] = None) -> pd.DataFrame:
        """Parse all downloaded transcripts into a single corpus.

        Args:
            paths: PDF paths; defaults to every PDF in the output dir.

        Returns:
            Concatenated speaker-turn DataFrame across all meetings.
        """
        if paths is None:
            paths = sorted(self.output_dir.glob("FOMCpresconf*.pdf"))
        frames = []
        for path in paths:
            frame = self.parse_transcript(path)
            if frame.empty:
                logger.warning("Empty parse for %s", path.name)
            else:
                frames.append(frame)
        corpus = pd.concat(frames, ignore_index=True)
        logger.info(
            "Parsed %d transcripts into %d speaker turns (%d Chair turns)",
            len(frames), len(corpus), (corpus["role"] == "chair").sum(),
        )
        return corpus


def main() -> None:
    """CLI entry point: download and parse the press conference corpus."""
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="data/processed/fomc_pressconf_turns.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    scraper = FomcPressConfScraper(config)
    scraper.download()
    corpus = scraper.build_corpus()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(output, index=False)
    logger.info("Wrote %d turns to %s", len(corpus), output)


if __name__ == "__main__":
    main()
