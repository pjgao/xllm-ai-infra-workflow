#!/usr/bin/env python3
"""Query xLLM model PR optimization history.

Searches across model archive files for PR history, optimization attempts,
known risks, and pending ideas related to specific models.
"""

import argparse
import os
import re
import sys
from pathlib import Path


def load_archives(archives_dir: str) -> dict[str, str]:
    archives = {}
    archive_path = Path(archives_dir)
    if not archive_path.exists():
        return archives
    for md_file in archive_path.glob("*.md"):
        archives[md_file.stem] = md_file.read_text()
    return archives


def search_archives(archives: dict[str, str], query: str) -> list[dict]:
    results = []
    query_lower = query.lower()
    patterns = query_lower.split()

    for model, content in archives.items():
        content_lower = content.lower()
        match_count = sum(1 for p in patterns if p in content_lower)
        if match_count > 0:
            sections = extract_relevant_sections(content, patterns)
            results.append({
                "model": model,
                "match_count": match_count,
                "sections": sections,
            })

    results.sort(key=lambda r: -r["match_count"])
    return results


def extract_relevant_sections(content: str, patterns: list[str]) -> list[str]:
    sections = []
    lines = content.split("\n")
    current_section = []
    current_header = ""

    for line in lines:
        if line.startswith("## "):
            if current_section and any(p in "\n".join(current_section).lower() for p in patterns):
                sections.append(f"**{current_header}**\n" + "\n".join(current_section[-10:]))
            current_section = []
            current_header = line.strip("# ").strip()
        else:
            current_section.append(line)

    if current_section and any(p in "\n".join(current_section).lower() for p in patterns):
        sections.append(f"**{current_header}**\n" + "\n".join(current_section[-10:]))

    return sections


def format_results(results: list[dict], verbose: bool = False) -> str:
    if not results:
        return "No matching archives found."

    lines = [f"# PR Optimization History Search Results\n"]
    lines.append(f"Found {len(results)} matching model archives.\n")

    for r in results:
        lines.append(f"## {r['model']}")
        lines.append(f"Match score: {r['match_count']}\n")

        for i, section in enumerate(r["sections"][:5]):
            if verbose:
                lines.append(section)
            else:
                preview = section[:200] + "..." if len(section) > 200 else section
                lines.append(preview)
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query xLLM model PR optimization history")
    parser.add_argument("query", help="Search query (model name, keyword, etc.)")
    parser.add_argument("--archives-dir", default=os.path.join(os.path.dirname(__file__), "..", "xllm"),
                        help="Path to model archives directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full section content")
    args = parser.parse_args()

    archives = load_archives(args.archives_dir)
    if not archives:
        print(f"No archives found in {args.archives_dir}", file=sys.stderr)
        sys.exit(1)

    results = search_archives(archives, args.query)
    print(format_results(results, args.verbose))


if __name__ == "__main__":
    main()
