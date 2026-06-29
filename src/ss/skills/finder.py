"""SkillFinder: best-effort skill discovery via the `npx skills` CLI.

Discovery shells out to `npx skills add <source> --list`, which lists the skills
in a `vercel-labs/skills`-compatible source repository (each a `SKILL.md` folder
with `name` + `description`) **without installing them**. The CLI's `find`
command is interactive (fzf-style) and not scriptable, so `add --list` against a
configured source is the programmatic path.

Best-effort by design: any failure — `npx`/node missing, timeout, non-zero exit,
or unparseable output — yields an empty list and never blocks the pipeline. The
listing is cached per source for the finder's lifetime, so a session runs the
CLI at most once per source regardless of how many strategies query it.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from ss.config import Config

logger = logging.getLogger(__name__)

# `add --list` clones the source from GitHub, so allow more than a trivial timeout.
_LIST_TIMEOUT = 45.0
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
# Clack renders a left gutter of box/spinner glyphs; strip them plus leading space.
_GUTTER_RE = re.compile(r"^[\s│◇◆◯●◒◐◓◑○◌◦·•>─-]+")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_STOPWORDS = frozenset({
    "the", "a", "an", "to", "of", "and", "or", "for", "in", "on", "with", "is",
    "are", "be", "this", "that", "use", "using", "when", "via", "from",
})


@dataclass
class FoundSkill:
    """A skill discovered in a `vercel-labs/skills`-compatible source repository."""
    name: str
    description: str
    source: str


class SkillFinder:
    """Discovers reusable skills via the `npx skills` CLI (best-effort, cached)."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._cache: dict[str, list[FoundSkill]] = {}

    async def find(self, capability_description: str) -> list[FoundSkill]:
        """Return skills from the configured source, ranked by relevance to the query.

        Returns ``[]`` (never raises) when disabled, when the CLI is unavailable
        or times out, or when no output can be parsed.
        """
        if not self._config.findskills_enabled:
            return []
        source = self._config.findskills_source
        if source not in self._cache:
            output = await self._run_list(source)
            self._cache[source] = self._parse_list(output, source) if output else []
        return self._rank(
            self._cache[source], capability_description, self._config.findskills_max_results
        )

    async def _run_list(self, source: str) -> str | None:
        """Run `npx skills add <source> --list` and return its stdout, or None."""
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "npx", "--yes", "skills@latest", "add", source, "--list", "--yes",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=_LIST_TIMEOUT)
            return out.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            logger.warning("`npx skills add %s --list` timed out; skipping discovery", source)
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            return None
        except (FileNotFoundError, OSError) as exc:
            logger.warning("skills CLI unavailable (%s); skipping skill discovery", exc)
            return None

    @staticmethod
    def _parse_list(text: str, source: str) -> list[FoundSkill]:
        """Parse the CLI's 'Available Skills' section into (name, description) pairs.

        After the 'Available Skills' header, each skill is a kebab-case name line
        followed by a prose description line, rendered inside a Clack box gutter.
        """
        skills: list[FoundSkill] = []
        seen_header = False
        pending: str | None = None
        for raw in text.splitlines():
            line = _GUTTER_RE.sub("", _ANSI_RE.sub("", raw)).strip()
            if not line:
                continue
            if "Available Skills" in line:
                seen_header = True
                continue
            if not seen_header:
                continue
            if " " not in line and _SLUG_RE.match(line):
                pending = line  # a skill name (kebab slug)
            elif pending is not None:
                skills.append(FoundSkill(name=pending, description=line, source=source))
                pending = None
        return skills

    @staticmethod
    def _rank(skills: list[FoundSkill], query: str, limit: int) -> list[FoundSkill]:
        """Rank skills by query-term overlap; surface a few even if nothing matches."""
        terms = {
            t for t in re.findall(r"[a-z0-9]+", query.lower())
            if len(t) > 2 and t not in _STOPWORDS
        }
        if not terms:
            return skills[:limit]

        def score(s: FoundSkill) -> int:
            haystack = f"{s.name} {s.description}".lower()
            return sum(1 for t in terms if t in haystack)

        ranked = sorted(skills, key=score, reverse=True)
        matched = [s for s in ranked if score(s) > 0]
        return (matched or ranked)[:limit]
