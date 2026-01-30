"""Generate synthetic sample data for offline pipeline testing.

Creates realistic central bank speech excerpts with controlled topic
distributions and sentiment, allowing the full pipeline to run without
web scraping.  Uses template-based text generation with keyword
injection to simulate different monetary policy themes.

Usage:
    python data/sample/generate_sample_data.py
"""

from __future__ import annotations

import csv
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Seed for reproducibility
RANDOM_SEED = 42

# ── Topic Templates ──────────────────────────────────────────────────────────
# Each topic has a set of paragraph templates and characteristic phrases.
# The generator selects templates probabilistically to create documents
# with known topic assignments for evaluation.

TOPIC_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "inflation_outlook": {
        "phrases": [
            "consumer price inflation", "inflationary pressures",
            "price stability mandate", "core inflation",
            "headline CPI", "inflation expectations",
            "supply-side price pressures", "cost-push factors",
            "demand-pull inflation", "second-round effects",
        ],
        "paragraphs": [
            (
                "Inflation has {trend} over the past quarter, with headline CPI "
                "{direction} to {rate}%. Core inflation, which strips out volatile "
                "food and energy prices, has {core_trend}. We continue to monitor "
                "inflation expectations closely, as their anchoring is critical "
                "for the credibility of our monetary policy framework."
            ),
            (
                "The persistence of inflationary pressures reflects a combination "
                "of supply-side constraints and robust domestic demand. While "
                "energy prices have {energy_trend}, services inflation remains "
                "{services_level}. The pass-through from producer prices to "
                "consumer prices continues to evolve, and we expect {outlook} "
                "in the coming months."
            ),
            (
                "Our projections indicate that inflation will {projection} over "
                "the medium term, converging towards our {target}% target by "
                "{horizon}. However, risks to this forecast are tilted to the "
                "{risk_direction}, reflecting uncertainty around global commodity "
                "markets and domestic wage dynamics."
            ),
        ],
    },
    "interest_rate_policy": {
        "phrases": [
            "policy rate", "interest rate decision", "monetary stance",
            "rate-setting", "forward guidance", "terminal rate",
            "neutral rate", "rate cycle", "tightening cycle",
            "restrictive territory",
        ],
        "paragraphs": [
            (
                "The Committee voted to {decision} the Bank Rate at {rate}%. "
                "This decision reflects our assessment that the current monetary "
                "policy stance is {stance}. We will continue to set policy to "
                "ensure inflation returns sustainably to target, while being "
                "mindful of the impact on economic activity and employment."
            ),
            (
                "The transmission of our previous rate {changes} to the real "
                "economy is still working through. Mortgage rates have "
                "{mortgage_trend}, and credit conditions have {credit_trend}. "
                "The full impact of cumulative tightening is expected to "
                "materialise over the coming quarters."
            ),
            (
                "Looking at the path of interest rates, market pricing suggests "
                "expectations of {market_expectation}. Our forward guidance "
                "remains data-dependent: we will assess the evolving outlook "
                "at each meeting and adjust the policy stance as warranted by "
                "incoming information on inflation and economic activity."
            ),
        ],
    },
    "economic_growth": {
        "phrases": [
            "GDP growth", "economic output", "recession risk",
            "business investment", "productivity growth",
            "aggregate demand", "output gap", "potential growth",
            "economic resilience", "sectoral performance",
        ],
        "paragraphs": [
            (
                "The economy grew by {rate}% in the latest quarter, "
                "{comparison} market expectations. Business investment has "
                "{investment_trend}, supported by {investment_factor}. "
                "The services sector continues to {services_trend}, while "
                "manufacturing output has {manufacturing_trend}."
            ),
            (
                "Household consumption remains {consumption_state}, reflecting "
                "the balance between real income growth and the drag from higher "
                "borrowing costs. Consumer confidence has {confidence_trend} "
                "in recent months. The housing market shows signs of "
                "{housing_state}."
            ),
            (
                "Our central projection is for GDP growth of {forecast}% in "
                "{year}, with risks skewed to the {risk_direction}. Potential "
                "output growth is estimated at around {potential}%, reflecting "
                "structural factors including demographics, capital deepening, "
                "and total factor productivity trends."
            ),
        ],
    },
    "labour_market": {
        "phrases": [
            "unemployment rate", "labour market tightness",
            "wage growth", "employment growth", "labour force participation",
            "vacancies", "skills shortage", "wage-price spiral",
            "real wages", "unit labour costs",
        ],
        "paragraphs": [
            (
                "The labour market remains {state}, with the unemployment rate "
                "at {rate}%. Employment has {employment_trend} by {jobs}k over "
                "the quarter. Vacancies have {vacancy_trend} but remain "
                "{vacancy_level} by historical standards."
            ),
            (
                "Wage growth has {wage_trend} to {wage_rate}% in annual terms. "
                "Private sector regular pay growth of {private_wage}% continues "
                "to exceed rates consistent with the inflation target, given "
                "current productivity trends. We are monitoring closely for "
                "signs of a wage-price spiral."
            ),
            (
                "Labour force participation has {participation_trend}, "
                "partly reflecting {participation_factor}. Skills shortages "
                "persist in {shortage_sectors}, contributing to wage pressures "
                "in these areas. The balance of demand and supply in the labour "
                "market remains a key determinant of the inflation outlook."
            ),
        ],
    },
    "financial_stability": {
        "phrases": [
            "financial conditions", "systemic risk", "banking sector",
            "credit growth", "asset valuations", "leverage",
            "macroprudential", "stress testing", "capital buffers",
            "non-bank financial institutions",
        ],
        "paragraphs": [
            (
                "Financial conditions have {trend} since our last assessment. "
                "Equity markets have {equity_trend}, while corporate bond "
                "spreads have {spread_trend}. Overall, financial conditions "
                "remain {overall_state} for the current stage of the cycle."
            ),
            (
                "The banking sector continues to exhibit {bank_state} capital "
                "and liquidity positions. Aggregate Common Equity Tier 1 ratios "
                "stand at {cet1}%, well above regulatory minima. Credit growth "
                "has {credit_trend} to {credit_rate}%, reflecting both supply "
                "and demand factors."
            ),
            (
                "Risks from non-bank financial institutions warrant continued "
                "monitoring. {nbfi_concern}. Our macroprudential framework "
                "provides tools to address emerging vulnerabilities, and we "
                "stand ready to adjust countercyclical buffers if conditions "
                "warrant."
            ),
        ],
    },
    "global_risks": {
        "phrases": [
            "geopolitical tensions", "global trade", "supply chain",
            "emerging markets", "commodity prices", "exchange rate",
            "cross-border spillovers", "global growth",
            "trade policy uncertainty", "energy security",
        ],
        "paragraphs": [
            (
                "The global economic environment remains {state}. Growth in "
                "major trading partners has {growth_trend}, with divergent "
                "paths across regions. {region} continues to {region_trend}, "
                "while {other_region} faces {challenge}."
            ),
            (
                "Geopolitical risks have {geo_trend} since our previous "
                "assessment. {geo_specific}. These developments pose risks to "
                "global supply chains, energy markets, and trade flows. The "
                "exchange rate has {fx_trend} by {fx_change}% on a trade-"
                "weighted basis."
            ),
            (
                "Commodity prices, particularly {commodity}, have {price_trend}. "
                "This reflects {price_factor}. The impact on domestic inflation "
                "will depend on the persistence of these price movements and "
                "the degree of pass-through to consumer prices."
            ),
        ],
    },
}

# Fill-in values for template slots
FILL_VALUES: Dict[str, List[str]] = {
    "trend": ["risen", "fallen", "remained broadly stable", "moderated"],
    "direction": ["rising", "falling", "edging up", "edging down"],
    "rate": ["2.1", "3.4", "4.7", "5.2", "6.1", "2.8", "3.9"],
    "core_trend": ["remained sticky", "shown signs of easing", "edged higher"],
    "energy_trend": ["stabilised", "declined further", "remained volatile"],
    "services_level": ["elevated", "sticky", "above target-consistent levels"],
    "outlook": ["gradual disinflation", "continued moderation", "further easing"],
    "projection": ["decline gradually", "remain elevated before easing", "converge"],
    "target": ["2"],
    "horizon": ["late 2025", "mid-2026", "the end of 2026"],
    "risk_direction": ["upside", "downside"],
    "decision": ["maintain", "increase", "reduce"],
    "stance": ["appropriately restrictive", "sufficiently tight", "balanced"],
    "changes": ["increases", "adjustments"],
    "mortgage_trend": ["risen further", "stabilised", "begun to ease"],
    "credit_trend": ["tightened", "remained tight", "shown tentative loosening"],
    "market_expectation": [
        "a gradual easing over the coming year",
        "rates remaining at current levels for an extended period",
        "one further increase followed by a pause",
    ],
    "comparison": ["broadly in line with", "slightly above", "slightly below"],
    "investment_trend": ["picked up modestly", "remained subdued", "recovered"],
    "investment_factor": [
        "improved business sentiment", "public sector spending",
        "infrastructure investment",
    ],
    "services_trend": ["expand steadily", "show resilience", "moderate"],
    "manufacturing_trend": ["contracted", "stabilised", "shown tentative signs of recovery"],
    "consumption_state": ["resilient", "subdued", "mixed"],
    "confidence_trend": ["improved", "deteriorated", "remained flat"],
    "housing_state": ["stabilisation", "gradual recovery", "continued adjustment"],
    "forecast": ["0.5", "1.0", "1.4", "0.8", "1.2"],
    "year": ["2025", "2026"],
    "potential": ["1.0", "1.2", "1.5"],
    "state": ["tight", "gradually loosening", "resilient", "uncertain"],
    "employment_trend": ["increased", "remained broadly flat", "declined"],
    "jobs": ["50", "100", "150", "75", "200"],
    "vacancy_trend": ["declined", "stabilised", "continued to fall"],
    "vacancy_level": ["elevated", "above pre-pandemic levels", "near historical norms"],
    "wage_trend": ["moderated", "remained elevated", "accelerated"],
    "wage_rate": ["5.2", "6.1", "4.8", "7.0", "5.5"],
    "private_wage": ["5.5", "6.3", "4.9"],
    "participation_trend": ["recovered", "remained below pre-pandemic levels", "stabilised"],
    "participation_factor": [
        "long-term sickness", "early retirement trends",
        "increased economic inactivity among older workers",
    ],
    "shortage_sectors": [
        "healthcare and technology", "construction and engineering",
        "hospitality and logistics",
    ],
    "equity_trend": ["risen", "declined", "remained range-bound"],
    "spread_trend": ["widened", "narrowed", "remained stable"],
    "overall_state": ["accommodative", "neutral", "restrictive"],
    "bank_state": ["robust", "adequate", "strong"],
    "cet1": ["14.2", "15.1", "13.8"],
    "credit_rate": ["2.5", "1.8", "3.2"],
    "nbfi_concern": [
        "Leverage in certain hedge fund strategies remains elevated",
        "Liquidity mismatches in open-ended funds require attention",
        "Growth in private credit markets warrants close monitoring",
    ],
    "growth_trend": ["slowed", "remained resilient", "diverged"],
    "region": ["The United States", "China", "The euro area"],
    "region_trend": ["grow above trend", "face headwinds", "recover gradually"],
    "other_region": ["emerging Asia", "Latin America", "sub-Saharan Africa"],
    "challenge": ["capital outflow pressures", "fiscal consolidation needs", "commodity headwinds"],
    "geo_trend": ["intensified", "remained elevated", "evolved"],
    "geo_specific": [
        "Tensions in the Middle East continue to affect energy markets",
        "Trade policy uncertainty has risen following recent tariff announcements",
        "The conflict in Eastern Europe continues to weigh on European growth",
    ],
    "fx_trend": ["appreciated", "depreciated", "remained broadly stable"],
    "fx_change": ["1.2", "2.5", "0.8", "3.1"],
    "commodity": ["oil", "natural gas", "agricultural commodities"],
    "price_trend": ["risen sharply", "fallen", "remained volatile"],
    "price_factor": [
        "supply disruptions", "weakening global demand",
        "geopolitical risk premia", "seasonal factors",
    ],
}


class SampleDataGenerator:
    """Generates synthetic central bank speech data for offline testing.

    Creates realistic-looking documents with known topic assignments
    and controlled sentiment characteristics, enabling the full
    pipeline to run without requiring web scraping.

    Attributes:
        n_documents: Number of documents to generate.
        start_date: Earliest document date.
        end_date: Latest document date.
        output_dir: Directory to save generated data.
    """

    def __init__(
        self,
        n_documents: int = 200,
        start_date: str = "2015-01-01",
        end_date: str = "2025-06-30",
        output_dir: str = "data/sample",
    ) -> None:
        """Initialise the sample data generator.

        Args:
            n_documents: Number of documents to generate.
            start_date: Start of the date range (YYYY-MM-DD).
            end_date: End of the date range (YYYY-MM-DD).
            output_dir: Directory to save output files.
        """
        self.n_documents = n_documents
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> pd.DataFrame:
        """Generate all sample documents.

        Returns:
            DataFrame with columns: date, title, speaker, source,
            doc_type, text, url, primary_topic.
        """
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        topics = list(TOPIC_TEMPLATES.keys())
        sources = ["boe", "fed", "ecb"]
        speakers = {
            "boe": [
                "Andrew Bailey", "Ben Broadbent", "Sarah Breeden",
                "Huw Pill", "Silvana Tenreyro", "Dave Ramsden",
            ],
            "fed": [
                "Jerome Powell", "Lael Brainard", "Christopher Waller",
                "Michelle Bowman", "Philip Jefferson", "Lisa Cook",
            ],
            "ecb": [
                "Christine Lagarde", "Philip Lane", "Isabel Schnabel",
                "Fabio Panetta", "Frank Elderson", "Piero Cipollone",
            ],
        }
        doc_types = ["speech", "speech", "speech", "minutes"]  # 75% speeches

        # Generate evenly spaced dates with jitter
        date_range = (self.end_date - self.start_date).days
        base_dates = [
            self.start_date + timedelta(days=int(i * date_range / self.n_documents))
            for i in range(self.n_documents)
        ]
        # Add random jitter of ±3 days
        dates = [
            d + timedelta(days=random.randint(-3, 3))
            for d in base_dates
        ]
        dates = [max(self.start_date, min(self.end_date, d)) for d in dates]

        records: List[Dict[str, str]] = []
        for i, date in enumerate(dates):
            source = sources[i % 3]
            speaker = random.choice(speakers[source])
            doc_type = random.choice(doc_types)
            primary_topic = random.choice(topics)

            # Generate text with 2-3 paragraphs from primary topic
            # and 0-1 paragraphs from a secondary topic
            n_primary = random.randint(2, 3)
            n_secondary = random.randint(0, 1)
            secondary_topic = random.choice(
                [t for t in topics if t != primary_topic]
            )

            paragraphs = []
            for _ in range(n_primary):
                template = random.choice(TOPIC_TEMPLATES[primary_topic]["paragraphs"])
                paragraphs.append(self._fill_template(template))

            for _ in range(n_secondary):
                template = random.choice(TOPIC_TEMPLATES[secondary_topic]["paragraphs"])
                paragraphs.append(self._fill_template(template))

            text = "\n\n".join(paragraphs)
            title = self._generate_title(primary_topic, speaker, date)

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "title": title,
                "speaker": speaker if doc_type == "speech" else "Committee",
                "source": source,
                "doc_type": doc_type,
                "text": text,
                "url": f"https://example.com/{source}/{date.strftime('%Y%m%d')}-{i}",
                "primary_topic": primary_topic,
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        logger.info("Generated %d sample documents", len(df))
        return df

    def _fill_template(self, template: str) -> str:
        """Fill a paragraph template with random values.

        Args:
            template: Template string with {placeholder} slots.

        Returns:
            Filled paragraph text.
        """
        import re as _re

        placeholders = _re.findall(r"\{(\w+)\}", template)
        values = {}
        for ph in placeholders:
            if ph in FILL_VALUES:
                values[ph] = random.choice(FILL_VALUES[ph])
            else:
                values[ph] = ph  # Keep unknown placeholders as-is
        return template.format(**values)

    @staticmethod
    def _generate_title(topic: str, speaker: str, date: datetime) -> str:
        """Generate a realistic speech title.

        Args:
            topic: Primary topic key.
            speaker: Speaker name.
            date: Document date.

        Returns:
            Title string.
        """
        title_templates = {
            "inflation_outlook": [
                "Inflation dynamics and the path ahead",
                "Price stability in a changing world",
                "The inflation outlook: challenges and opportunities",
            ],
            "interest_rate_policy": [
                "Monetary policy in the current environment",
                "The path of interest rates",
                "Setting policy in uncertain times",
            ],
            "economic_growth": [
                "The economic outlook",
                "Growth prospects and structural change",
                "Navigating the economic cycle",
            ],
            "labour_market": [
                "Labour market dynamics and wages",
                "Employment, wages, and inflation",
                "The changing nature of work",
            ],
            "financial_stability": [
                "Financial stability and resilience",
                "The stability of the financial system",
                "Macroprudential policy and financial conditions",
            ],
            "global_risks": [
                "Global risks and domestic implications",
                "The international economic outlook",
                "Navigating global uncertainty",
            ],
        }
        templates = title_templates.get(topic, ["Economic outlook and policy"])
        return random.choice(templates)

    def save(self, df: pd.DataFrame) -> Path:
        """Save the generated sample speeches to CSV.

        Args:
            df: Generated sample DataFrame.

        Returns:
            Path to the saved CSV file.
        """
        output_path = self.output_dir / "sample_speeches.csv"
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        logger.info("Saved %d sample speeches to %s", len(df), output_path)
        return output_path

    def load_or_generate(self) -> pd.DataFrame:
        """Load existing sample data or generate new data if not present.

        Returns:
            DataFrame of sample speeches.
        """
        csv_path = self.output_dir / "sample_speeches.csv"
        if csv_path.exists():
            logger.info("Loading existing sample data from %s", csv_path)
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            return df

        logger.info("Generating new sample data")
        df = self.generate()
        self.save(df)
        return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    generator = SampleDataGenerator()
    df = generator.generate()
    generator.save(df)
    print(f"Generated {len(df)} sample documents")
    print(f"Topics: {df['primary_topic'].value_counts().to_dict()}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
