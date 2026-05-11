from __future__ import annotations

from typing import List

from playwright.sync_api import sync_playwright


def scrape_cli_explorer_paths(url: str = "https://apps.juniper.net/cli-explorer/") -> List[list[str]]:
    """Extract hierarchical CLI paths from dynamic CLI Explorer SPA.

    This implementation relies on generic ARIA tree roles and is resilient to minor DOM shifts.
    """
    paths: List[list[str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        tree_items = page.locator('[role="treeitem"]')
        count = tree_items.count()
        for idx in range(count):
            node = tree_items.nth(idx)
            try:
                expanded = node.get_attribute("aria-expanded")
                if expanded == "false":
                    node.click()
            except Exception:
                continue

        labels = page.locator('[role="treeitem"]')
        for i in range(labels.count()):
            label = labels.nth(i).inner_text().strip()
            if not label:
                continue
            parts = [p.strip() for p in label.split("/") if p.strip()]
            if len(parts) >= 2:
                paths.append(parts)

        browser.close()
    return paths
