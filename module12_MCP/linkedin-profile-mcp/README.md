# LinkedIn Profile Scraper MCP Server

This MCP server uses the Fresh LinkedIn Profile Data API to fetch LinkedIn profile information. It is implemented as a model context protocol (MCP) server and exposes a single tool, `get_profile`, which accepts a LinkedIn profile URL and returns the profile data in JSON format.

## Features

- **Fetch Profile Data:** Retrieves LinkedIn profile information including skills and other settings (with most additional details disabled).
- **Asynchronous HTTP Requests:** Uses `httpx` for non-blocking API calls.
- **Environment-based Configuration:** Reads the `RAPIDAPI_KEY` from your environment variables using `dotenv`.

## Prerequisites

- **Python 3.7+** – Ensure you are using Python version 3.7 or higher.
- **MCP Framework:** Make sure the MCP framework is installed.
- **Required Libraries:** Install `httpx`, `python-dotenv`, and other dependencies.
- **RAPIDAPI_KEY:** Obtain an API key from [RapidAPI](https://rapidapi.com/) and add it to a `.env` file in your project directory (or set it in your environment).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/AIAnytime/Awesome-MCP-Server
   cd linkedin_profile_scraper
   ```

2. **Install Dependencies:**

   ```bash
   uv add mcp[cli] httpx requests
   ```


3. **Set Up Environment Variables:**

   Create a `.env` file in the project directory with the following content:

   ```ini
   RAPIDAPI_KEY=your_rapidapi_key_here
   ```

## Running the Server

To run the MCP server, execute:

```bash
uv run linkedin.py
```

The server will start and listen for incoming requests via standard I/O.

## MCP Client Configuration

To connect your MCP client to this server, add the following configuration to your `config.json`. Adjust the paths as necessary for your environment:

```json
{
  "mcpServers": {
    "linkedin_profile_scraper": {
      "command": "C:/Users/aiany/.local/bin/uv",
      "args": [
        "--directory",
        "C:/Users/aiany/OneDrive/Desktop/YT Video/linkedin-mcp/project",
        "run",
        "linkedin.py"
      ]
    }
  }
}
```

## Code Overview

- **Environment Setup:** The server uses `dotenv` to load the `RAPIDAPI_KEY` required to authenticate with the Fresh LinkedIn Profile Data API.
- **API Call:** The asynchronous function `get_linkedin_data` makes a GET request to the API with specified query parameters.
- **MCP Tool:** The `get_profile` tool wraps the API call and returns formatted JSON data, or an error message if the call fails.
- **Server Execution:** The MCP server is run with the `stdio` transport.

## Troubleshooting

- **Missing RAPIDAPI_KEY:** If the key is not set, the server will raise a `ValueError`. Make sure the key is added to your `.env` file or set in your environment.
- **API Errors:** If the API request fails, the tool will return a message indicating that the profile data could not be fetched.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
