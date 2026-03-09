from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class WebHit:
    title: str
    snippet: str
    url: str
    source: str


class WebSearcher:
    def __init__(self) -> None:
        mode = os.getenv("ALICE_WEB_SEARCH_MODE", "auto").strip().lower()
        if mode not in {"off", "auto", "always"}:
            mode = "auto"
        self._mode = mode
        timeout_raw = os.getenv("ALICE_WEB_SEARCH_TIMEOUT", "6")
        try:
            self._timeout = max(2.0, min(15.0, float(timeout_raw)))
        except ValueError:
            self._timeout = 6.0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def enabled(self) -> bool:
        return self._mode != "off"

    def should_search(self, text: str) -> bool:
        if not self.enabled:
            return False
        if self._mode == "always":
            return True

        lowered = text.strip().lower()
        if not lowered:
            return False

        starts = (
            "what ",
            "who ",
            "when ",
            "where ",
            "why ",
            "how ",
            "is ",
            "are ",
            "do ",
            "does ",
            "can ",
            "tell me about",
            "explain",
        )
        temporal = (
            "today",
            "latest",
            "recent",
            "news",
            "price",
            "stock",
            "weather",
            "update",
            "current",
            "right now",
        )
        return lowered.endswith("?") or lowered.startswith(starts) or any(
            token in lowered for token in temporal
        )

    def lookup(self, query: str, *, max_results: int = 3) -> list[WebHit]:
        if not self.enabled:
            return []
        trimmed = query.strip()
        if not trimmed:
            return []

        hits: list[WebHit] = []
        for variant in self._query_variants(trimmed):
            if len(hits) < max_results:
                hits.extend(self._duckduckgo_hits(variant, max_results=max_results - len(hits)))
            if len(hits) < max_results:
                hits.extend(self._wikipedia_hits(variant, max_results=max_results - len(hits)))
            if len(hits) >= max_results:
                break

        deduped: list[WebHit] = []
        seen: set[str] = set()
        for hit in hits:
            key = (hit.url or hit.title).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
            if len(deduped) >= max_results:
                break
        return deduped

    def _query_variants(self, query: str) -> list[str]:
        cleaned = query.strip()
        if not cleaned:
            return []
        simplified = cleaned.lower().strip()
        simplified = re.sub(r"[?!.]+$", "", simplified)
        simplified = re.sub(
            r"^(alice[,:\s-]*)?",
            "",
            simplified,
            flags=re.IGNORECASE,
        )
        simplified = re.sub(
            r"^(do you know|can you tell me|please tell me|could you explain|what is|what's|who is|tell me about|explain)\s+",
            "",
            simplified,
            flags=re.IGNORECASE,
        )
        simplified = simplified.strip()
        what_x_is = re.match(r"^what\s+(.+)\s+is$", simplified)
        if what_x_is:
            simplified = f"what is {what_x_is.group(1).strip()}"
        variants = [cleaned]
        if simplified and simplified != cleaned.lower():
            variants.append(simplified)
        if simplified.startswith("what is "):
            topic = simplified.removeprefix("what is ").strip()
            if topic:
                variants.append(topic)
        return variants

    def format_for_prompt(self, hits: list[WebHit]) -> list[str]:
        lines: list[str] = []
        for hit in hits:
            lines.append(
                f"{hit.title}: {hit.snippet} (source: {hit.source}, url: {hit.url})"
            )
        return lines

    def _get_json(self, url: str) -> dict:
        raw = self._fetch_text(url)
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}

    def _fetch_text(self, url: str) -> str:
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "User-Agent": "AliceAssistant/1.0",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return response.read().decode("utf-8")
        except Exception:
            result = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "--fail",
                    "-H",
                    "User-Agent: AliceAssistant/1.0",
                    "-H",
                    "Accept: application/json",
                    url,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout + 3.0,
            )
            if result.returncode != 0 or not result.stdout:
                raise RuntimeError("web fetch failed")
            return result.stdout

    def _duckduckgo_hits(self, query: str, *, max_results: int) -> list[WebHit]:
        try:
            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                }
            )
            data = self._get_json(f"https://api.duckduckgo.com/?{params}")
        except Exception:
            return []

        hits: list[WebHit] = []

        answer = str(data.get("Answer", "")).strip()
        answer_type = str(data.get("AnswerType", "")).strip()
        if answer:
            hits.append(
                WebHit(
                    title=answer_type or "Answer",
                    snippet=answer,
                    url=str(data.get("AbstractURL", "")).strip(),
                    source="duckduckgo",
                )
            )

        abstract = str(data.get("AbstractText", "")).strip() or str(data.get("Abstract", "")).strip()
        if abstract:
            hits.append(
                WebHit(
                    title=str(data.get("Heading", "")).strip() or query,
                    snippet=abstract,
                    url=str(data.get("AbstractURL", "")).strip(),
                    source="duckduckgo",
                )
            )

        definition = str(data.get("Definition", "")).strip()
        if definition:
            hits.append(
                WebHit(
                    title=str(data.get("Heading", "")).strip() or query,
                    snippet=definition,
                    url=str(data.get("DefinitionURL", "")).strip(),
                    source="duckduckgo",
                )
            )

        related = data.get("RelatedTopics", [])
        if isinstance(related, list):
            for item in related:
                if len(hits) >= max_results:
                    break
                if not isinstance(item, dict):
                    continue
                nested = item.get("Topics")
                if isinstance(nested, list):
                    for nested_item in nested:
                        if len(hits) >= max_results:
                            break
                        hit = self._related_topic_to_hit(nested_item)
                        if hit is not None:
                            hits.append(hit)
                    continue
                hit = self._related_topic_to_hit(item)
                if hit is not None:
                    hits.append(hit)

        return hits[:max_results]

    def _related_topic_to_hit(self, item: object) -> WebHit | None:
        if not isinstance(item, dict):
            return None
        text = str(item.get("Text", "")).strip()
        url = str(item.get("FirstURL", "")).strip()
        if not text:
            return None
        title = text.split(" - ", 1)[0].strip()
        return WebHit(
            title=title or "Related",
            snippet=text,
            url=url,
            source="duckduckgo",
        )

    def _wikipedia_hits(self, query: str, *, max_results: int) -> list[WebHit]:
        try:
            params = urllib.parse.urlencode(
                {
                    "action": "opensearch",
                    "search": query,
                    "limit": max_results,
                    "namespace": "0",
                    "format": "json",
                }
            )
            raw = self._fetch_text(f"https://en.wikipedia.org/w/api.php?{params}")
            payload = json.loads(raw)
        except Exception:
            return []

        if not isinstance(payload, list) or len(payload) < 2:
            return []
        titles = payload[1]
        if not isinstance(titles, list):
            return []

        hits: list[WebHit] = []
        for title_obj in titles[:max_results]:
            title = str(title_obj).strip()
            if not title:
                continue
            encoded_title = urllib.parse.quote(title, safe="")
            try:
                summary = self._get_json(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
                )
            except Exception:
                continue

            snippet = str(summary.get("extract", "")).strip()
            page_url = ""
            content_urls = summary.get("content_urls", {})
            if isinstance(content_urls, dict):
                desktop = content_urls.get("desktop", {})
                if isinstance(desktop, dict):
                    page_url = str(desktop.get("page", "")).strip()
            if not snippet:
                continue
            hits.append(
                WebHit(
                    title=title,
                    snippet=snippet,
                    url=page_url or f"https://en.wikipedia.org/wiki/{encoded_title}",
                    source="wikipedia",
                )
            )
        return hits
