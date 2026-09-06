import re

def extract_video_id(url: str) -> str | None:
    """
    Extracts the YouTube video ID from various URL formats.
    Returns the 11-character video ID as a string, or None if not found.
    """
    if not url or not url.strip():
        return None

    url = url.strip()

    # Covers:
    # - youtube.com/watch?v=VIDEO_ID
    # - youtu.be/VIDEO_ID
    # - youtube.com/embed/VIDEO_ID
    # - youtube.com/shorts/VIDEO_ID
    patterns = [
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def is_valid_youtube_url(url: str) -> bool:
    """
    Quick check: does this URL contain a valid-looking YouTube video ID?
    """
    return extract_video_id(url) is not None


import requests

def get_video_metadata(video_id: str) -> dict | None:
    """
    Fetches basic video metadata (title, author, thumbnail) using
    YouTube's public oEmbed endpoint. No API key required.
    Returns a dict, or None if the video doesn't exist / isn't accessible.
    """
    oembed_url = "https://www.youtube.com/oembed"
    params = {
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "format": "json"
    }

    try:
        response = requests.get(oembed_url, params=params, timeout=10)
        response.raise_for_status()  # raises an error if status isn't 200 OK
        data = response.json()

        return {
            "title": data.get("title", "Unknown Title"),
            "channel": data.get("author_name", "Unknown Channel"),
            "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        }
    except requests.exceptions.RequestException:
        # Covers: video doesn't exist, is private, network error, etc.
        return None