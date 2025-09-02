from mcp.server.fastmcp import FastMCP
from Bio import Entrez
import time
import warnings
import asyncio

# Suppress warnings and set email for Entrez
warnings.filterwarnings('ignore')
Entrez.email = "your email"

# Initialize FastMCP server
mcp = FastMCP("pubmed")

def fetch_pubmed_articles(query: str = "endocarditis", max_results: int = 20) -> list[str]:
    """Fetch PubMed articles for the given query without saving to a file."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()

    ids = record.get('IdList', [])
    articles = []
    for pmid in ids:
        time.sleep(0.5)  # Delay to avoid overwhelming the API
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
            abstract = handle.read()
            handle.close()
            if abstract:
                articles.append(abstract.strip())
        except Exception:
            continue
    return articles

@mcp.tool()
async def search_pubmed(query: str = "endocarditis", max_results: int = 10) -> str:
    """
    Search PubMed for articles matching the query.

    Args:
        query: The search term for PubMed.
        max_results: Maximum number of articles to retrieve.

    Returns:
        A string containing the abstracts of found articles, separated by two newlines.
    """
    # Run the blocking function in a separate thread
    articles = await asyncio.to_thread(fetch_pubmed_articles, query, max_results)
    if articles:
        return "\n\n".join(articles)
    else:
        return "No articles found for the given query."

if __name__ == "__main__":
    mcp.run(transport='stdio')
