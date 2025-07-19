# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
#     "typer",
#     "pyyaml",
# ]
# ///
"""
Flight search and filtering module for finding business class award seats.
"""

import itertools
import json
import os
import pickle
import smtplib
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
import typer
import yaml
from typing_extensions import Annotated


@dataclass
class SearchParams:
    """Parameters for flight search."""

    origin_airport: str
    destination_airport: str
    start_date: str
    end_date: str
    cabin_class: str = "business"
    include_filtered: bool = True
    include_trips: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        return {
            "origin_airport": self.origin_airport,
            "destination_airport": self.destination_airport,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "cabin_class": self.cabin_class,
            "include_filtered": self.include_filtered,
            "include_trips": self.include_trips,
        }


@dataclass
class FlightResult:
    """Structured flight result data."""

    source: str
    mileage_cost: int
    remaining_seats: int
    date: str
    origin: str
    destination: str
    is_direct: bool
    airlines: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SearchConfig:
    """Configuration for a flight search."""

    name: str
    origins: List[str]
    destinations: List[str]
    start_date: str
    end_date: str
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchConfig":
        """Create SearchConfig from dictionary."""
        return cls(
            name=data["name"],
            origins=data["origins"],
            destinations=data["destinations"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            description=data.get("description", data["name"]),
        )


@dataclass
class EmailConfig:
    """Email configuration settings."""

    smtp_server: str
    smtp_port: int
    sender_email: str
    sender_password: str
    recipient_emails: List[str]
    use_tls: bool = True


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""

    data: Dict[str, Any]
    timestamp: float
    ttl: int  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > self.ttl

    @classmethod
    def create(cls, data: Dict[str, Any], ttl: int = 3600) -> "CacheEntry":
        """Create a new cache entry with current timestamp."""
        return cls(data=data, timestamp=time.time(), ttl=ttl)


class PersistentTTLCache:
    """File-based cache with TTL expiration."""

    def __init__(self, cache_file: Path, default_ttl: int = 3600):
        """
        Initialize persistent cache.

        Args:
            cache_file: Path to cache file
            default_ttl: Default TTL in seconds (1 hour by default)
        """
        self.cache_file = cache_file
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                # Clean up expired entries on load
                self._cleanup_expired()
        except (pickle.PickleError, EOFError, OSError) as e:
            print(f"Warning: Could not load cache file {self.cache_file}: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except (pickle.PickleError, OSError) as e:
            print(f"Warning: Could not save cache file {self.cache_file}: {e}")

    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        self._cleanup_expired()

        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                return entry.data
            else:
                # Remove expired entry
                del self._cache[key]

        return None

    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl

        self._cache[key] = CacheEntry.create(value, ttl)
        self._save_cache()

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()

    def size(self) -> int:
        """Get number of entries in cache."""
        self._cleanup_expired()
        return len(self._cache)

    def cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._cleanup_expired()
        total_size = sum(len(str(entry.data)) for entry in self._cache.values())
        return {
            "entries": len(self._cache),
            "cache_file": str(self.cache_file),
            "approximate_size_bytes": total_size,
            "default_ttl_seconds": self.default_ttl,
        }


@dataclass
class SearchSummary:
    """Summary of search results for JSON output."""

    search_name: str
    search_description: str
    origins: List[str]
    destinations: List[str]
    start_date: str
    end_date: str
    results_count: int
    results: List[FlightResult]
    search_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "search_name": self.search_name,
            "search_description": self.search_description,
            "origins": self.origins,
            "destinations": self.destinations,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "results_count": self.results_count,
            "results": [result.to_dict() for result in self.results],
            "search_timestamp": self.search_timestamp,
        }


class FlightSearcher:
    """Flight search and filtering service."""

    # Airport codes
    US_AIRPORTS = ["sfo", "lax", "sea", "iah", "dfw", "ont"]
    ASIA_AIRPORTS = ["hkg", "tpe", "icn", "hnd", "nrt", "pvg", "pek", "wuh"]

    # Filter criteria
    MAX_MILEAGE_COST = 140000
    PREFERRED_SOURCES = {"united", "alaska", "aeroplan"}

    def __init__(
        self,
        email_config: Optional[EmailConfig] = None,
        cache_file: Optional[Path] = None,
        cache_ttl: int = 3600,  # 1 hour default
    ):
        """
        Initialize with API credentials from environment and optional email configuration.

        Args:
            email_config: Email configuration for notifications
            cache_file: Path to cache file (defaults to .flight_cache.pkl)
            cache_ttl: Cache TTL in seconds (default: 1 hour)
        """
        partner_auth_token = os.environ.get("SEATS_AERO_PARTNER_AUTH", "")
        self.api_url = "https://seats.aero/partnerapi/search"
        self.headers = {
            "accept": "application/json",
            "Partner-Authorization": partner_auth_token,
        }
        self.email_config = email_config

        # Initialize persistent cache
        if cache_file is None:
            cache_file = Path.cwd() / ".flight_cache.pkl"
        self.cache = PersistentTTLCache(cache_file, cache_ttl)

    def _fetch_flights(self, params_json: str) -> Dict[str, Any]:
        """
        Fetch flight data from API with persistent caching and TTL.

        Args:
            params_json: JSON string of search parameters

        Returns:
            API response as dictionary
        """
        # Use params_json as cache key
        cache_key = f"flight_search:{hash(params_json)}"

        # Try to get from cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            print(f"🗂️  Using cached result for search parameters")
            return cached_result

        # Not in cache or expired, fetch from API
        params = json.loads(params_json)

        try:
            print(f"🌐 Fetching from API...")
            response = requests.get(
                self.api_url, headers=self.headers, params=params, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # Cache the result
            self.cache.set(cache_key, result)
            return result

        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return {"data": []}

    def clear_cache(self) -> None:
        """Clear all cached flight data."""
        self.cache.clear()
        print("✅ Cache cleared successfully")

    def cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return self.cache.cache_info()

    def _meets_criteria(self, flight_data: Dict[str, Any]) -> bool:
        """
        Check if flight meets filtering criteria.

        Args:
            flight_data: Single flight record from API

        Returns:
            True if flight meets all criteria
        """
        return (
            flight_data.get("JAvailable", False)
            and flight_data.get("JMileageCostRaw", float("inf")) < self.MAX_MILEAGE_COST
            and flight_data.get("JRemainingSeats", 0) > 0
            and flight_data.get("Source", "") in self.PREFERRED_SOURCES
        )

    def _extract_flight_info(self, flight_data: Dict[str, Any]) -> FlightResult:
        """
        Extract relevant flight information into structured format.

        Args:
            flight_data: Single flight record from API

        Returns:
            FlightResult object with extracted data
        """
        route = flight_data.get("Route", {})

        return FlightResult(
            source=flight_data.get("Source", ""),
            mileage_cost=flight_data.get("JMileageCostRaw", 0),
            remaining_seats=flight_data.get("JRemainingSeats", 0),
            date=flight_data.get("Date", ""),
            origin=route.get("OriginAirport", ""),
            destination=route.get("DestinationAirport", ""),
            is_direct=flight_data.get("JDirect", False),
            airlines=flight_data.get("JAirlines", ""),
        )

    def search_and_filter(
        self, search_params: SearchParams
    ) -> Generator[FlightResult, None, None]:
        """
        Search for flights and yield those meeting criteria.

        Args:
            search_params: Search parameters

        Yields:
            FlightResult objects for qualifying flights
        """
        params_json = json.dumps(search_params.to_dict())
        response = self._fetch_flights(params_json)

        for flight_data in response.get("data", []):
            if self._meets_criteria(flight_data):
                yield self._extract_flight_info(flight_data)

    def search_route_combinations(
        self,
        origins: List[str],
        destinations: List[str],
        start_date: str,
        end_date: str,
    ) -> Generator[FlightResult, None, None]:
        """
        Search all combinations of origin/destination airports.

        Args:
            origins: List of origin airport codes
            destinations: List of destination airport codes
            start_date: Search start date (YYYY-MM-DD)
            end_date: Search end date (YYYY-MM-DD)

        Yields:
            FlightResult objects for all qualifying flights
        """
        for origin, destination in itertools.product(origins, destinations):
            search_params = SearchParams(
                origin_airport=origin,
                destination_airport=destination,
                start_date=start_date,
                end_date=end_date,
            )

            yield from self.search_and_filter(search_params)

    def format_result(self, result: FlightResult) -> str:
        """Format flight result for display."""
        return (
            f"{result.source} | {result.mileage_cost:,} miles | "
            f"{result.remaining_seats} seats | {result.date} | "
            f"{result.origin.upper()} → {result.destination.upper()} | "
            f"{'Direct' if result.is_direct else 'Connecting'} | {result.airlines}"
        )

    def _create_email_content(
        self, results: List[FlightResult], search_description: str
    ) -> Tuple[str, str]:
        """
        Create email subject and HTML content from flight results.

        Args:
            results: List of flight results
            search_description: Description of the search performed

        Returns:
            Tuple of (subject, html_content)
        """
        if not results:
            subject = f"Flight Search Results - No flights found ({search_description})"
            html_content = f"""
            <html>
            <body>
                <h2>Flight Search Results</h2>
                <p><strong>Search:</strong> {search_description}</p>
                <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>No flights found matching your criteria.</p>
            </body>
            </html>
            """
        else:
            subject = f"Flight Search Results - {len(results)} flights found ({search_description})"

            # Create HTML table
            table_rows = ""
            for result in results:
                table_rows += f"""
                <tr>
                    <td>{result.source}</td>
                    <td>{result.mileage_cost:,}</td>
                    <td>{result.remaining_seats}</td>
                    <td>{result.date}</td>
                    <td>{result.origin.upper()}</td>
                    <td>{result.destination.upper()}</td>
                    <td>{'Direct' if result.is_direct else 'Connecting'}</td>
                    <td>{result.airlines}</td>
                </tr>
                """

            html_content = f"""
            <html>
            <head>
                <style>
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .summary {{ background-color: #e7f3ff; padding: 10px; margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                <h2>Flight Search Results</h2>
                <div class="summary">
                    <p><strong>Search:</strong> {search_description}</p>
                    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Results Found:</strong> {len(results)} flights</p>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>Source</th>
                            <th>Miles</th>
                            <th>Seats</th>
                            <th>Date</th>
                            <th>Origin</th>
                            <th>Destination</th>
                            <th>Type</th>
                            <th>Airlines</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </body>
            </html>
            """

        return subject, html_content

    def send_email(self, results: List[FlightResult], search_description: str) -> bool:
        """
        Send flight results via email.

        Args:
            results: List of flight results to send
            search_description: Description of the search performed

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.email_config:
            print("Email configuration not provided. Cannot send email.")
            return False

        try:
            # Create email content
            subject, html_content = self._create_email_content(
                results, search_description
            )

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.email_config.sender_email
            msg["To"] = ", ".join(self.email_config.recipient_emails)

            # Add HTML content
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(
                self.email_config.smtp_server, self.email_config.smtp_port
            ) as server:
                if self.email_config.use_tls:
                    server.starttls()
                server.login(
                    self.email_config.sender_email, self.email_config.sender_password
                )
                server.send_message(msg)

            print(
                f"Email sent successfully to {', '.join(self.email_config.recipient_emails)}"
            )
            return True

        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    def search_and_email(
        self,
        origins: List[str],
        destinations: List[str],
        start_date: str,
        end_date: str,
        search_description: str,
    ) -> List[FlightResult]:
        """
        Search for flights and automatically email the results.

        Args:
            origins: List of origin airport codes
            destinations: List of destination airport codes
            start_date: Search start date (YYYY-MM-DD)
            end_date: Search end date (YYYY-MM-DD)
            search_description: Description for email subject/content

        Returns:
            List of flight results found
        """
        # Collect all results
        results = list(
            self.search_route_combinations(origins, destinations, start_date, end_date)
        )

        # Send email if configured
        if self.email_config and results:
            self.send_email(results, search_description)

        return results

    def save_results_to_json(
        self,
        search_summaries: List[SearchSummary],
        output_path: Path,
        pretty_print: bool = True,
    ) -> bool:
        """
        Save search results to JSON file.

        Args:
            search_summaries: List of search summaries to save
            output_path: Path to output JSON file
            pretty_print: Whether to format JSON with indentation

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create output data structure
            output_data = {
                "search_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_searches": len(search_summaries),
                    "total_flights_found": sum(
                        summary.results_count for summary in search_summaries
                    ),
                },
                "searches": [summary.to_dict() for summary in search_summaries],
            }

            # Write to file
            with open(output_path, "w") as f:
                if pretty_print:
                    json.dump(output_data, f, indent=2, default=str)
                else:
                    json.dump(output_data, f, default=str)

            return True

        except Exception as e:
            print(f"Failed to save JSON file: {e}")
            return False


# CLI Application
app = typer.Typer(help="Search for business class award flights")


def create_email_config(
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    recipient_emails: List[str],
    use_tls: bool = True,
) -> Optional[EmailConfig]:
    """Create email configuration if password is available."""
    sender_password = os.environ.get("EMAIL_APP_PASSWORD", "")
    if not sender_password:
        typer.echo(
            "Warning: EMAIL_APP_PASSWORD environment variable not set. Email disabled."
        )
        return None

    return EmailConfig(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        sender_email=sender_email,
        sender_password=sender_password,
        recipient_emails=recipient_emails,
        use_tls=use_tls,
    )


def load_search_config(config_path: Path) -> List[SearchConfig]:
    """Load search configurations from YAML file."""
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        searches = []
        for search_data in data.get("searches", []):
            searches.append(SearchConfig.from_dict(search_data))

        return searches

    except FileNotFoundError:
        typer.echo(f"❌ Configuration file not found: {config_path}")
        raise typer.Exit(1)
    except yaml.YAMLError as e:
        typer.echo(f"❌ Error parsing YAML file: {e}")
        raise typer.Exit(1)
    except KeyError as e:
        typer.echo(f"❌ Missing required field in configuration: {e}")
        raise typer.Exit(1)


def create_sample_config(config_path: Path) -> None:
    """Create a sample YAML configuration file."""
    sample_config = {
        "searches": [
            {
                "name": "US to Asia Thanksgiving",
                "origins": ["sfo", "lax", "sea"],
                "destinations": ["nrt", "hnd", "icn"],
                "start_date": "2025-11-20",
                "end_date": "2025-11-25",
                "description": "Thanksgiving trip to Asia",
            },
            {
                "name": "Asia to US Return",
                "origins": ["nrt", "hnd", "icn"],
                "destinations": ["sfo", "lax", "sea"],
                "start_date": "2025-12-01",
                "end_date": "2025-12-07",
                "description": "Return from Asia after Thanksgiving",
            },
            {
                "name": "West Coast to Europe",
                "origins": ["sfo", "lax"],
                "destinations": ["fra", "lhr", "cdg"],
                "start_date": "2025-06-01",
                "end_date": "2025-06-15",
                "description": "Summer Europe trip",
            },
        ]
    }

    with open(config_path, "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)

    typer.echo(f"✅ Sample configuration created at: {config_path}")


@app.command()
def init_config(
    config_path: Annotated[
        Path, typer.Option("--config", help="Path to configuration file")
    ] = Path("flight_search.yaml"),
):
    """Create a sample YAML configuration file."""
    if config_path.exists():
        overwrite = typer.confirm(
            f"Configuration file {config_path} already exists. Overwrite?"
        )
        if not overwrite:
            typer.echo("Operation cancelled.")
            raise typer.Exit()

    create_sample_config(config_path)

    typer.echo("\nSample configuration structure:")
    typer.echo("searches:")
    typer.echo("  - name: 'Search Name'")
    typer.echo("    origins: ['sfo', 'lax']")
    typer.echo("    destinations: ['nrt', 'hnd']")
    typer.echo("    start_date: '2025-11-15'")
    typer.echo("    end_date: '2025-11-20'")
    typer.echo("    description: 'Optional description'")


@app.command()
def clear_cache(
    cache_file: Annotated[
        Optional[Path], typer.Option("--cache-file", help="Path to cache file")
    ] = None,
):
    """Clear the flight search cache."""
    if cache_file is None:
        cache_file = Path.cwd() / ".flight_cache.pkl"

    cache = PersistentTTLCache(cache_file)
    cache.clear()
    typer.echo(f"✅ Cache cleared: {cache_file}")


@app.command()
def cache_info(
    cache_file: Annotated[
        Optional[Path], typer.Option("--cache-file", help="Path to cache file")
    ] = None,
):
    """Show cache information and statistics."""
    if cache_file is None:
        cache_file = Path.cwd() / ".flight_cache.pkl"

    cache = PersistentTTLCache(cache_file)
    info = cache.cache_info()

    typer.echo("📊 Cache Information")
    typer.echo("=" * 50)
    typer.echo(f"Cache file: {info['cache_file']}")
    typer.echo(f"Entries: {info['entries']}")
    typer.echo(f"Approximate size: {info['approximate_size_bytes']:,} bytes")
    typer.echo(
        f"Default TTL: {info['default_ttl_seconds']} seconds ({info['default_ttl_seconds']//3600}h {(info['default_ttl_seconds']%3600)//60}m)"
    )

    if info["entries"] == 0:
        typer.echo("🗂️  Cache is empty")
    else:
        typer.echo(f"🗂️  Cache contains {info['entries']} entries")


@app.command()
def search(
    config_path: Annotated[
        Path, typer.Option("--config", help="Path to YAML configuration file")
    ] = Path("flight_search.yaml"),
    search_name: Annotated[
        Optional[str],
        typer.Option(
            "--search", help="Specific search name to run (runs all if not specified)"
        ),
    ] = None,
    send_email: Annotated[
        bool, typer.Option("--email/--no-email", help="Send results via email")
    ] = False,
    sender_email: Annotated[
        str, typer.Option("--sender-email", help="Sender email address")
    ] = "lianjunj@gmail.com",
    recipient_emails: Annotated[
        List[str],
        typer.Option(
            "--recipient", help="Recipient email address (can be used multiple times)"
        ),
    ] = None,
    smtp_server: Annotated[
        str, typer.Option("--smtp-server", help="SMTP server")
    ] = "smtp.gmail.com",
    smtp_port: Annotated[int, typer.Option("--smtp-port", help="SMTP port")] = 587,
    list_searches: Annotated[
        bool, typer.Option("--list", help="List available searches in config file")
    ] = False,
    json_output: Annotated[
        Optional[Path],
        typer.Option("--json-output", "-j", help="Save results to JSON file"),
    ] = None,
    pretty_json: Annotated[
        bool,
        typer.Option(
            "--pretty-json/--compact-json", help="Format JSON with indentation"
        ),
    ] = True,
    cache_file: Annotated[
        Optional[Path], typer.Option("--cache-file", help="Path to cache file")
    ] = None,
    cache_ttl: Annotated[
        int,
        typer.Option(
            "--cache-ttl", help="Cache TTL in seconds (default: 3600 = 1 hour)"
        ),
    ] = 3600,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Disable caching for this search")
    ] = False,
):
    """Run flight searches based on YAML configuration file."""

    if not config_path.exists():
        typer.echo(f"❌ Configuration file not found: {config_path}")
        typer.echo(
            "💡 Run 'python script.py init-config' to create a sample configuration."
        )
        raise typer.Exit(1)

    # Load search configurations
    search_configs = load_search_config(config_path)

    if list_searches:
        typer.echo(f"📋 Available searches in {config_path}:")
        for config in search_configs:
            typer.echo(
                f"  • {config.name}: {', '.join(config.origins)} → {', '.join(config.destinations)} ({config.start_date} to {config.end_date})"
            )
        return

    if recipient_emails is None:
        recipient_emails = ["lianjunj@gmail.com"]

    # Create email configuration if requested
    email_config = None
    if send_email:
        email_config = create_email_config(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            sender_email=sender_email,
            recipient_emails=recipient_emails,
        )
        if not email_config:
            typer.echo(
                "Email requested but configuration failed. Continuing without email."
            )

    # Filter searches if specific search requested
    if search_name:
        search_configs = [
            config for config in search_configs if config.name == search_name
        ]
        if not search_configs:
            typer.echo(f"❌ Search '{search_name}' not found in configuration.")
            typer.echo("💡 Use --list to see available searches.")
            raise typer.Exit(1)

    # Initialize searcher with cache configuration
    if no_cache:
        # Use a dummy cache file that won't be used
        cache_file = Path("/dev/null") if cache_file is None else cache_file
        cache_ttl = 0  # Immediate expiration
    elif cache_file is None:
        cache_file = Path.cwd() / ".flight_cache.pkl"

    searcher = FlightSearcher(
        email_config=email_config, cache_file=cache_file, cache_ttl=cache_ttl
    )

    # Show cache info if requested
    if not no_cache:
        cache_info = searcher.cache_info()
        if cache_info["entries"] > 0:
            typer.echo(
                f"🗂️  Using cache with {cache_info['entries']} entries (TTL: {cache_ttl}s)"
            )
        else:
            typer.echo(f"🗂️  Cache initialized (TTL: {cache_ttl}s)")

    all_results = []
    search_summaries = []

    for config in search_configs:
        typer.echo(f"\n🔍 Running search: {config.name}")
        typer.echo(
            f"   Routes: {', '.join(config.origins)} → {', '.join(config.destinations)}"
        )
        typer.echo(f"   Dates: {config.start_date} to {config.end_date}")
        typer.echo("=" * 80)

        # Search flights
        if email_config:
            results = searcher.search_and_email(
                origins=config.origins,
                destinations=config.destinations,
                start_date=config.start_date,
                end_date=config.end_date,
                search_description=config.description or config.name,
            )
        else:
            results = list(
                searcher.search_route_combinations(
                    origins=config.origins,
                    destinations=config.destinations,
                    start_date=config.start_date,
                    end_date=config.end_date,
                )
            )

        # Create search summary for JSON output
        search_summary = SearchSummary(
            search_name=config.name,
            search_description=config.description or config.name,
            origins=config.origins,
            destinations=config.destinations,
            start_date=config.start_date,
            end_date=config.end_date,
            results_count=len(results),
            results=results,
            search_timestamp=datetime.now().isoformat(),
        )
        search_summaries.append(search_summary)

        # Print results to console
        if results:
            for result in results:
                typer.echo(searcher.format_result(result))
            typer.echo(f"✅ Found {len(results)} flights for '{config.name}'")
        else:
            typer.echo(f"❌ No flights found for '{config.name}'")

        all_results.extend(results)

    # Save to JSON if requested
    if json_output:
        success = searcher.save_results_to_json(
            search_summaries, json_output, pretty_json
        )
        if success:
            typer.echo(f"💾 Results saved to JSON file: {json_output}")
        else:
            typer.echo(f"❌ Failed to save JSON file: {json_output}")

    # Summary
    typer.echo(f"\n📊 SUMMARY")
    typer.echo("=" * 80)
    typer.echo(f"Total searches run: {len(search_configs)}")
    typer.echo(f"Total flights found: {len(all_results)}")

    if email_config:
        typer.echo("✅ Results have been emailed to configured recipients.")
    elif send_email:
        typer.echo("⚠️  Email was requested but not sent due to configuration issues.")
    else:
        typer.echo("📧 Email not requested - results displayed in console only.")

    if json_output:
        typer.echo(f"📄 Results saved to: {json_output}")


if __name__ == "__main__":
    app()
