"""
Daily UK Legislation Scraper
----------------------------

This module scrapes newly published legislation from legislation.gov.uk
in a safe, automated way. It supports:

• Text-based legislation pages
• PDF legislation items (using PyMuPDF)
• Pagination across long multi-page legislation
• Consistent folder structure for storing results
• Structured JSON metadata for downstream retrieval tasks

This scraper is used to supplement the statutory retrieval benchmark
released with the paper: "Legal Reranking for Statutory Retrieval".

Author: Amal (Durham University)
"""

import os
import time
import tempfile
import requests
import pymupdf
from selenium import webdriver
from selenium.webdriver.common.by import By


class DailyScraper:
    """Scraper for daily updates on legislation.gov.uk."""

    BASE_URLS = {
        "UK": "https://www.legislation.gov.uk/new/uk",
        "Wales": "https://www.legislation.gov.uk/new/wales",
        "Scotland": "https://www.legislation.gov.uk/new/scotland",
        "NorthernIreland": "https://www.legislation.gov.uk/new/ni",
    }

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.scraped_dir = os.path.join(base_path, "Scraped_Content")
        self.new_content_dir = os.path.join(base_path, "New_Content")
        self.current_year = time.gmtime().tm_year

    # -------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------

    def _ensure_dir(self, path: str):
        """Create directory if it does not exist."""
        os.makedirs(path, exist_ok=True)

    def _fix_filename(self, name: str) -> str:
        """Sanitize filenames to avoid slashes."""
        return name.replace("/", "_").replace("\\", "_")

    # -------------------------------------------------------
    # Verification
    # -------------------------------------------------------

    def _has_new_update(self, driver) -> bool:
        """Check if the update page lists new legislation."""
        header = driver.find_element(By.CLASS_NAME, "p_content").find_element(By.TAG_NAME, "h5").text
        return header.strip() != "Nothing published on this date"

    # -------------------------------------------------------
    # PDF extraction
    # -------------------------------------------------------

    def _extract_pdf(self, url: str) -> str:
        """Download & extract text from a PDF legislation file."""
        resp = requests.get(url)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        doc = pymupdf.open(tmp_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

        return f"\nURL_OF_PAGE:\n{url}\n{text}"

    # -------------------------------------------------------
    # Text extraction
    # -------------------------------------------------------

    def _extract_text_content(self, driver, content_url: str) -> str:
        """Extract multi-page text legislation."""
        driver.get(content_url)
        time.sleep(1)

        all_text = f"\nURL_OF_PAGE:\n{content_url}\n"
        page_number = 1

        while True:
            content_box = driver.find_element(By.ID, "content")
            snippet = content_box.find_element(By.ID, "viewLegContents").find_element(By.CLASS_NAME, "LegSnippet")
            all_text += snippet.text

            print(f"[Scraper] Page {page_number}")

            # Next button?
            try:
                next_link = (
                    driver.find_element(By.CLASS_NAME, "prevNextNav")
                    .find_element(By.TAG_NAME, "ul")
                    .find_elements(By.TAG_NAME, "li")[-1]
                    .find_element(By.TAG_NAME, "a")
                )
                next_url = next_link.get_attribute("href")
                next_link.click()
                page_number += 1
                time.sleep(1.5)
                all_text += f"\nURL_OF_PAGE:\n{next_url}\n"
            except Exception:
                print("[Scraper] Reached final page.")
                break

        return all_text

    # -------------------------------------------------------
    # Extract a single legislation item
    # -------------------------------------------------------

    def extract_item(self, driver, item_url: str) -> str:
        """
        Extract full text or PDF content from a legislation item page.
        """

        driver.get(item_url)
        time.sleep(1)

        try:
            toc = driver.find_element(By.CSS_SELECTOR, "div.legToc")
            nav = toc.find_element(By.ID, "legSubNav")
            tabs = nav.find_elements(By.TAG_NAME, "li")

            # Tab[1] = "Content"
            content_tab = tabs[1]

            try:
                link = content_tab.find_element(By.TAG_NAME, "a")
                return self._extract_text_content(driver, link.get_attribute("href"))
            except Exception:
                # Not text → PDF
                pdf_link = (
                    driver.find_element(By.CSS_SELECTOR, "div.LegSnippet")
                    .find_element(By.TAG_NAME, "a")
                    .get_attribute("href")
                )
                return self._extract_pdf(pdf_link)
        except Exception as e:
            print(f"[Error] Could not extract item: {item_url}")
            print(e)
            return None

    # -------------------------------------------------------
    # Parse a daily update page
    # -------------------------------------------------------

    def get_daily_titles(self, driver, url: str):
        """Return dict of legislation_name → {title: href} for new items."""
        driver.get(url)
        time.sleep(1)

        if not self._has_new_update(driver):
            return None

        container = driver.find_element(By.CLASS_NAME, "p_content")
        legislation_name = container.find_element(By.TAG_NAME, "h5").text.strip()
        h6_items = container.find_elements(By.TAG_NAME, "h6")

        titles = {}
        for h6 in h6_items:
            title_name = h6.text.split("-")[-1].strip()
            href = h6.find_element(By.TAG_NAME, "a").get_attribute("href")
            titles[title_name] = href

        return {legislation_name: titles}

    # -------------------------------------------------------
    # Main driver
    # -------------------------------------------------------

    def run(self):
        """
        Main scraper driver. Yields dicts:

        {
            "Text": "...",
            "Meta": {
                "Country": "...",
                "LegislationType": "...",
                "Legislation": "...",
                "Year": "2024",
                "Title": "...",
            }
        }
        """

        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(options=options)

        for country, url in self.BASE_URLS.items():
            print(f"[Scraper] Checking updates for {country}...")
            updates = self.get_daily_titles(driver, url)

            if not updates:
                continue

            for legislation_name, titles in updates.items():
                for title, href in titles.items():

                    text = self.extract_item(driver, href)
                    if text is None:
                        continue

                    # Store text
                    safe_name = self._fix_filename(title)
                    out_dir = os.path.join(
                        self.new_content_dir, country, legislation_name, str(self.current_year)
                    )
                    self._ensure_dir(out_dir)

                    out_path = os.path.join(out_dir, f"{safe_name}.txt")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(text)

                    yield {
                        "Text": text,
                        "Meta": {
                            "Country": country,
                            "LegislationType": legislation_name,
                            "Legislation": legislation_name,
                            "Year": str(self.current_year),
                            "Title": title,
                        },
                    }

        driver.quit()
