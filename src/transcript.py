from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


def fetch_transcript(video_id: str) -> list[dict] | None:
    """
    Fetches the transcript for a YouTube video.
    Returns a list of dicts like:
        [{"text": "...", "start": 12.4, "duration": 3.2}, ...]
    Returns None if no transcript is available for this video.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)

        # fetched_transcript is an iterable of snippet objects;
        # to_raw_data() converts it into plain list-of-dicts form
        return fetched_transcript.to_raw_data()

    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video: {video_id}")
        return None
    except NoTranscriptFound:
        print(f"No transcript found for video: {video_id}")
        return None
    except VideoUnavailable:
        print(f"Video unavailable: {video_id}")
        return None
    except Exception as e:
        # Catch-all so a weird edge case doesn't crash the whole app
        print(f"Unexpected error fetching transcript: {e}")
        return None