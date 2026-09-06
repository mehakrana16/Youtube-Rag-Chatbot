import re


def clean_segment_text(text: str) -> str:
    """
    Cleans a single transcript segment's text:
    - Removes bracketed noise like [Music], [Applause]
    - Collapses excess whitespace/newlines
    - Strips leading/trailing spaces
    """
    if not text:
        return ""

    # Remove [Music], [Applause], [Laughter], etc.
    text = re.sub(r"\[.*?\]", "", text)

    # Replace newlines/tabs with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_transcript(transcript: list[dict]) -> list[dict]:
    """
    Applies clean_segment_text to every segment in the transcript.
    Keeps start/duration untouched. Drops segments that become empty
    after cleaning (e.g. a segment that was JUST "[Music]").
    """
    cleaned = []
    for seg in transcript:
        cleaned_text = clean_segment_text(seg["text"])
        if cleaned_text:  # skip empty segments
            cleaned.append({
                "text": cleaned_text,
                "start": seg["start"],
                "duration": seg["duration"],
            })
    return cleaned


def transcript_to_text_with_offsets(transcript: list[dict]) -> tuple[str, list[dict]]:
    """
    Joins all segment text into one continuous string, while building
    an "offset map" that remembers which character range came from
    which timestamp. This is what lets us later trace a chunk of text
    back to a specific point in the video.

    Returns:
        full_text: the whole transcript as one string
        offset_map: list of {"start_char": int, "end_char": int, "start_time": float}
    """
    full_text = ""
    offset_map = []

    for seg in transcript:
        start_char = len(full_text)
        full_text += seg["text"] + " "
        end_char = len(full_text)

        offset_map.append({
            "start_char": start_char,
            "end_char": end_char,
            "start_time": seg["start"],
        })

    return full_text.strip(), offset_map

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_transcript(full_text: str, offset_map: list[dict],
                      chunk_size: int = 800, chunk_overlap: int = 150) -> list[dict]:
    """
    Splits the full transcript text into overlapping chunks, and tags
    each chunk with the approximate video timestamp it starts at
    (using the offset_map built earlier).

    Returns a list of dicts:
        [{"text": "...", "start_time": 12.4}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # split_text just returns raw text chunks - we still need to
    # figure out where each one starts in the original full_text
    text_chunks = splitter.split_text(full_text)

    chunks_with_timestamps = []
    search_from = 0

    for chunk_text in text_chunks:
        # Find where this chunk starts in the full text
        char_position = full_text.find(chunk_text, search_from)
        if char_position == -1:
            char_position = full_text.find(chunk_text)  # fallback: search from start

        start_time = _lookup_timestamp(char_position, offset_map)

        chunks_with_timestamps.append({
            "text": chunk_text,
            "start_time": start_time,
        })

        # Move search position forward a bit (accounting for overlap)
        search_from = max(0, char_position + len(chunk_text) - chunk_overlap)

    return chunks_with_timestamps


def _lookup_timestamp(char_position: int, offset_map: list[dict]) -> float:
    """
    Given a character position in the full transcript text, finds
    which original segment it falls into and returns that segment's
    start_time.
    """
    for entry in offset_map:
        if entry["start_char"] <= char_position < entry["end_char"]:
            return entry["start_time"]
    # Fallback: if position is past the end, return the last known timestamp
    return offset_map[-1]["start_time"] if offset_map else 0.0