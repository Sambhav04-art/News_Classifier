"""Text preprocessing utilities."""
import html
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s']")
_SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise a raw news string.

    Steps: decode HTML entities -> strip tags/URLs -> remove the literal
    backslashes found in AG News (e.g. ``\\band``) -> lowercase ->
    drop punctuation -> collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.replace("\\", " ")
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def combine_title_description(title: str, description: str) -> str:
    """Join title and description into a single text field.

    The title is repeated once so that its (usually more discriminative)
    words get more weight in the TF-IDF representation.
    """
    title = title or ""
    description = description or ""
    return f"{title}. {title}. {description}"
