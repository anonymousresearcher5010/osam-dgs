import logging
import requests


logger = logging.getLogger(__name__)


def setup_session(user_agent: str) -> requests.Session:
    """Initialize and return a Requests session with the given user agent,
    to work only with a single persistent HTTP connection."""
    s = requests.Session()
    if 'example' in user_agent:
        logger.warning(f'Contact email address is probably not set. Please use your contact in the user agent for transparency reasons. Your current user agent: {user_agent}')
    s.headers.update({'User-Agent': user_agent})
    return s


def fetch_html(session: requests.Session, url: str, timeout: float) -> str:
    """Fetch the HTML content of the given URL, using the given Requests session."""
    r: requests.Response = session.get(url, timeout=timeout)

    r.raise_for_status()

    # Ensure the response uses UTF-8 encoding
    r.encoding = 'utf-8'

    return r.text
