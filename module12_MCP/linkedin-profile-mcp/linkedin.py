from typing import Any
import httpx
import json
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv

load_dotenv()

# Read the RAPIDAPI_KEY from the environment variables
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
if not RAPIDAPI_KEY:
    raise ValueError("RAPIDAPI_KEY is not set in the environment variables")

mcp = FastMCP("linkedin_profile_scraper")

LINKEDIN_API_BASE = "https://fresh-linkedin-profile-data.p.rapidapi.com"
RAPIDAPI_HOST = "fresh-linkedin-profile-data.p.rapidapi.com"

async def get_linkedin_data(linkedin_url: str) -> dict[str, Any] | None:
    """Fetch LinkedIn profile data using the Fresh LinkedIn Profile Data API."""
    params = {
        "linkedin_url": linkedin_url,
        "include_skills": "true",
        "include_certifications": "false",
        "include_publications": "false",
        "include_honors": "false",
        "include_volunteers": "false",
        "include_projects": "false",
        "include_patents": "false",
        "include_courses": "false",
        "include_organizations": "false",
        "include_profile_status": "false",
        "include_company_public_url": "false"
    }
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{LINKEDIN_API_BASE}/get-linkedin-profile",
                headers=headers,
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

@mcp.tool()
async def get_profile(linkedin_url: str) -> str:
    """Get LinkedIn profile data for a given profile URL.

    Args:
        linkedin_url: The LinkedIn profile URL.
    """
    data = await get_linkedin_data(linkedin_url)
    if not data:
        return "Unable to fetch LinkedIn profile data."
    return json.dumps(data, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")
