"""
Sanctions and Entity Screening Module.

This module provides tools to query sanctions lists and entity databases.
Currently implemented as a stub/placeholder for future OpenSanctions integration.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SanctionedEntity(BaseModel):
    """Represents a sanctioned entity from various lists."""

    entity_id: str = Field(description="Unique identifier for the entity")
    name: str = Field(description="Primary name of the entity")
    entity_type: str = Field(description="Type: person, organization, vessel, aircraft")
    aliases: list[str] = Field(default_factory=list, description="Known aliases")
    countries: list[str] = Field(default_factory=list, description="Associated countries")
    sanctions_lists: list[str] = Field(default_factory=list, description="Lists the entity appears on")
    identifiers: dict[str, str] = Field(
        default_factory=dict, description="IDs like passport numbers, tax IDs"
    )
    last_updated: str = Field(description="Last update timestamp")
    match_score: float = Field(description="Search match confidence (0-1)")


class SanctionsResponse(BaseModel):
    """Response model for sanctions queries."""

    status: str = Field(description="Query status: 'success', 'error', or 'stub'")
    query_params: dict[str, Any] = Field(description="Parameters used in the query")
    match_count: int = Field(description="Number of matching entities")
    entities: list[SanctionedEntity] = Field(default_factory=list)
    error_message: str | None = Field(default=None, description="Error message if any")
    data_source: str = Field(default="OpenSanctions (stub)", description="Data source identifier")
    is_stub: bool = Field(default=True, description="Whether this is mock data")


# Mock data for demonstration purposes
MOCK_ENTITIES = [
    SanctionedEntity(
        entity_id="mock-001",
        name="Example Sanctioned Person",
        entity_type="person",
        aliases=["E. S. Person", "Example Person"],
        countries=["RU"],
        sanctions_lists=["OFAC SDN", "EU Consolidated"],
        identifiers={"passport": "XX1234567"},
        last_updated="2024-01-15T00:00:00Z",
        match_score=0.95,
    ),
    SanctionedEntity(
        entity_id="mock-002",
        name="Example Sanctioned Organization",
        entity_type="organization",
        aliases=["ESO", "Example Org"],
        countries=["RU", "BY"],
        sanctions_lists=["OFAC SDN", "UK Sanctions"],
        identifiers={"registration": "RU123456789"},
        last_updated="2024-02-20T00:00:00Z",
        match_score=0.88,
    ),
    SanctionedEntity(
        entity_id="mock-003",
        name="Sanctioned Vessel Example",
        entity_type="vessel",
        aliases=["S/V Example"],
        countries=["RU"],
        sanctions_lists=["OFAC SDN"],
        identifiers={"imo": "1234567", "mmsi": "123456789"},
        last_updated="2024-03-10T00:00:00Z",
        match_score=0.92,
    ),
]


async def get_sanctions_info(
    query: str,
    entity_type: str | None = None,
    countries: list[str] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search sanctions lists for entities matching the query.

    **NOTE: This is currently a STUB implementation returning mock data.**
    Full integration with OpenSanctions API is planned for future releases.

    This tool searches consolidated sanctions lists to identify:
    - Sanctioned individuals (politicians, oligarchs, military personnel)
    - Sanctioned organizations (companies, military units, government agencies)
    - Sanctioned vessels and aircraft
    - Associated identifiers and aliases

    Data sources (when fully implemented):
    - OFAC SDN List (US Treasury)
    - EU Consolidated Sanctions List
    - UK Sanctions List
    - UN Security Council Sanctions
    - OpenSanctions consolidated database

    Args:
        query: Search query - name, alias, or identifier to search for.
               Supports partial matching and fuzzy search.
        entity_type: Optional filter by entity type:
                     'person', 'organization', 'vessel', 'aircraft', or None for all.
        countries: Optional list of ISO country codes to filter by (e.g., ["RU", "BY"]).
        limit: Maximum number of results to return (1-100). Default is 20.

    Returns:
        A dictionary containing:
        - status: 'success', 'error', or 'stub' (currently always 'stub')
        - query_params: The parameters used for the query
        - match_count: Number of matching entities found
        - entities: List of sanctioned entities with details
        - is_stub: Boolean indicating this is mock data (currently True)
        - error_message: Error details if the query failed

    Example:
        >>> result = await get_sanctions_info(
        ...     query="Ivanov",
        ...     entity_type="person",
        ...     countries=["RU"]
        ... )
        >>> for entity in result['entities']:
        ...     print(f"{entity['name']} - {entity['sanctions_lists']}")

    Future Integration:
        To enable full functionality, set up OpenSanctions API access:
        1. Visit https://www.opensanctions.org/api/
        2. Obtain an API key
        3. Set OPENSANCTIONS_API_KEY environment variable
    """
    limit = max(1, min(100, limit))

    query_params = {
        "query": query,
        "entity_type": entity_type,
        "countries": countries,
        "limit": limit,
        "query_time": datetime.utcnow().isoformat(),
    }

    # Filter mock entities based on query parameters
    filtered_entities: list[SanctionedEntity] = []

    for entity in MOCK_ENTITIES:
        # Check if query matches name or aliases
        query_lower = query.lower()
        name_match = query_lower in entity.name.lower()
        alias_match = any(query_lower in alias.lower() for alias in entity.aliases)

        if not (name_match or alias_match):
            continue

        # Filter by entity type if specified
        if entity_type and entity.entity_type != entity_type:
            continue

        # Filter by countries if specified
        if countries:
            country_match = any(c.upper() in entity.countries for c in countries)
            if not country_match:
                continue

        filtered_entities.append(entity)

        if len(filtered_entities) >= limit:
            break

    # If no matches found with mock data, create a sample response
    if not filtered_entities and query:
        # Return empty results with a note
        return SanctionsResponse(
            status="stub",
            query_params=query_params,
            match_count=0,
            entities=[],
            error_message=None,
            is_stub=True,
            data_source="OpenSanctions (stub - no mock matches)",
        ).model_dump()

    return SanctionsResponse(
        status="stub",
        query_params=query_params,
        match_count=len(filtered_entities),
        entities=filtered_entities,
        error_message="This is STUB data. Real OpenSanctions integration pending.",
        is_stub=True,
    ).model_dump()


async def check_entity_sanctions(
    name: str,
    date_of_birth: str | None = None,
    nationality: str | None = None,
) -> dict[str, Any]:
    """
    Perform a compliance screening check on a specific entity.

    **NOTE: This is currently a STUB implementation.**

    This is a simplified interface for quick compliance checks, useful for:
    - Screening individuals before transactions
    - Due diligence on business partners
    - Verifying counterparties in financial transactions

    Args:
        name: Full name of the person or organization to check.
        date_of_birth: Optional date of birth (YYYY-MM-DD) for better matching.
        nationality: Optional ISO country code for nationality.

    Returns:
        A dictionary with screening results and risk assessment.

    Example:
        >>> result = await check_entity_sanctions(
        ...     name="John Smith",
        ...     nationality="US"
        ... )
        >>> print(f"Risk Level: {result['risk_level']}")
    """
    query_params = {
        "name": name,
        "date_of_birth": date_of_birth,
        "nationality": nationality,
        "query_time": datetime.utcnow().isoformat(),
    }

    # Stub response
    return {
        "status": "stub",
        "query_params": query_params,
        "is_stub": True,
        "screening_result": {
            "matches_found": 0,
            "risk_level": "unknown",
            "requires_review": False,
            "matched_lists": [],
        },
        "note": "This is STUB data. Real screening requires OpenSanctions API integration.",
        "data_source": "OpenSanctions (stub)",
    }

