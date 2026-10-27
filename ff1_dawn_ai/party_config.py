"""Canonical party requests for both Final Fantasy platforms.

Keep canonical names in metadata even when a game's UI imposes a shorter
display alias. GBA Dawn of Souls has six name cells; original NES FF1 has four.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartyMember:
    canonical_name: str
    display_name: str
    job: str


GBA_CUSTOM_PARTY = (
    PartyMember("Sif", "Sif", "Warrior"),
    PartyMember("Myth", "Myth", "Thief"),
    PartyMember("Milo", "Milo", "Monk"),
    PartyMember("Mustard", "Mustar", "Red Mage"),
)

NES_SECONDARY_PARTY = (
    PartyMember("Sif", "Sif", "Warrior"),
    PartyMember("Mustard", "Must", "White Mage"),
    PartyMember("Locke", "Lock", "Black Mage"),
    PartyMember("Milo", "Milo", "Monk"),
)

GBA_NAME_LENGTH = 6
NES_NAME_LENGTH = 4


def validate_party(party: tuple[PartyMember, ...], max_name_length: int) -> None:
    if len(party) != 4:
        raise ValueError("Final Fantasy party must contain exactly four members")
    for member in party:
        if not member.display_name or len(member.display_name) > max_name_length:
            raise ValueError(
                f"Display name {member.display_name!r} exceeds {max_name_length} characters"
            )
        if not member.display_name.isascii() or not member.display_name.isalpha():
            raise ValueError(f"Unsupported display name: {member.display_name!r}")


validate_party(GBA_CUSTOM_PARTY, GBA_NAME_LENGTH)
validate_party(NES_SECONDARY_PARTY, NES_NAME_LENGTH)
