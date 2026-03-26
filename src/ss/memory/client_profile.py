"""ClientProfileManager: get-or-create and update client profiles."""
from __future__ import annotations

from datetime import datetime, timezone

from ss.blackboard.models import ClientProfile
from ss.blackboard.repository import Repository


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ClientProfileManager:
    """Manages persistent client profiles."""

    def __init__(self, repo: Repository) -> None:
        self._repo = repo

    def get_or_create(self, client_id: str) -> ClientProfile:
        """Return an existing profile or create a new one for the client."""
        profile = self._repo.get_client_profile(client_id)
        if profile is None:
            profile = ClientProfile(client_id=client_id)
            self._repo.upsert_client_profile(profile)
        return profile

    def update_after_session(
        self,
        client_id: str,
        expertise_level: str | None = None,
        known_domains: list[str] | None = None,
        communication_style: str | None = None,
    ) -> ClientProfile:
        """Update a client profile after a session completes.

        - Merges ``known_domains`` (no duplicates)
        - Increments ``total_sessions``
        - Updates ``expertise_level`` and ``communication_style`` if provided
        - Updates ``last_seen_at``
        """
        profile = self.get_or_create(client_id)

        if expertise_level is not None:
            profile = profile.model_copy(update={"expertise_level": expertise_level})

        if communication_style is not None:
            profile = profile.model_copy(update={"communication_style": communication_style})

        if known_domains:
            merged = list(dict.fromkeys(profile.known_domains + known_domains))
            profile = profile.model_copy(update={"known_domains": merged})

        profile = profile.model_copy(update={
            "total_sessions": profile.total_sessions + 1,
            "last_seen_at": _now(),
        })

        self._repo.upsert_client_profile(profile)
        return profile
