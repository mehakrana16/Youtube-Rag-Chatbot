import streamlit as st
from src.youtube import extract_video_id, is_valid_youtube_url, get_video_metadata
from src.transcript import fetch_transcript
from src.text_processing import clean_transcript, transcript_to_text_with_offsets, chunk_transcript
from src.embeddings import embed_chunks
from src.vector_store import VectorStore, retrieve_relevant_chunks
from src.llm import answer_question

# Page configuration
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide"
)

# ---- Custom CSS: Dark theme with pink accents ----
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a0d14 100%);
}
h1 {
    background: linear-gradient(90deg, #ff4d94, #ff85c1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
h2, h3, h4 {
    color: #ff6fa8 !important;
}
p, span, label, .stMarkdown, div[data-testid="stMarkdownContainer"] {
    color: #f5f5f5 !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: #ff9ec7 !important;
    font-size: 16px;
}
.stTextInput > div > div > input {
    background-color: #1f1f1f;
    color: #ffffff;
    border: 2px solid #ff85c1;
    border-radius: 12px;
    padding: 10px;
}
.stTextInput > div > div > input:focus {
    border-color: #ff4d94;
    box-shadow: 0 0 10px rgba(255, 77, 148, 0.5);
}
.stTextInput > div > div > input::placeholder {
    color: #999999;
}
.stButton > button {
    background: linear-gradient(90deg, #ff4d94, #ff85c1);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 28px;
    font-weight: 600;
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 15px rgba(255, 77, 148, 0.4);
    color: white;
}
div[data-testid="stAlert"] p {
    color: #1a1a1a !important;
    font-weight: 500;
}
.streamlit-expanderHeader, summary {
    color: #ff9ec7 !important;
}
div[data-testid="stExpander"] p {
    color: #f5f5f5 !important;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.title("🎥 YouTube RAG Chatbot")
st.caption("✨ Ask questions about any YouTube video's content")

# ---- Initialize session_state keys (only runs once per session) ----
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "video_metadata" not in st.session_state:
    st.session_state.video_metadata = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None

# URL input box
youtube_url = st.text_input(
    "Paste a YouTube video URL:",
    placeholder="https://www.youtube.com/watch?v=..."
)

# Load Video button
if st.button("🚀 Load Video"):
    if not youtube_url:
        st.warning("⚠️ Please paste a YouTube URL first.")
    elif not is_valid_youtube_url(youtube_url):
        st.error("❌ That doesn't look like a valid YouTube URL. Please check and try again.")
    else:
        video_id = extract_video_id(youtube_url)
        with st.spinner("Fetching video info..."):
            metadata = get_video_metadata(video_id)

        if metadata is None:
            st.error("❌ Couldn't fetch this video. It may be private, deleted, or age-restricted.")
        else:
            with st.spinner("Fetching transcript..."):
                transcript = fetch_transcript(video_id)

            if transcript is None:
                st.error("❌ No transcript available for this video (captions may be disabled).")
            else:
                transcript = clean_transcript(transcript)
                full_text, offset_map = transcript_to_text_with_offsets(transcript)
                chunks = chunk_transcript(full_text, offset_map)

                with st.spinner("Generating embeddings & building search index..."):
                    embeddings = embed_chunks(chunks)
                    store = VectorStore(embedding_dim=embeddings.shape[1])
                    store.add(embeddings, chunks)

                # Save everything into session_state so it survives reruns
                st.session_state.vector_store = store
                st.session_state.video_metadata = metadata
                st.session_state.video_id = video_id

                st.success(f"✅ Video processed — {len(chunks)} chunks indexed and ready for questions!")

# ---- Show video card if a video is loaded ----
if st.session_state.video_metadata is not None:
    metadata = st.session_state.video_metadata
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(metadata["thumbnail_url"], use_container_width=True)
    with col2:
        st.subheader(metadata["title"])
        st.write(f"**Channel:** {metadata['channel']}")
        st.caption(f"Video ID: `{st.session_state.video_id}`")

    st.divider()

    # ---- Chat / question section ----
    st.subheader("💬 Ask a question about this video")
    user_question = st.text_input("Your question:", placeholder="e.g. What is the main topic discussed?")

    if st.button("Ask"):
        if not user_question:
            st.warning("⚠️ Please type a question first.")
        else:
            with st.spinner("Searching transcript..."):
                results = retrieve_relevant_chunks(
                    st.session_state.vector_store,
                    user_question,
                    top_k=4
                )

            with st.spinner("Thinking..."):
                answer = answer_question(user_question, results)

            st.markdown("### 🤖 Answer")
            st.write(answer)

            with st.expander("📌 Sources used for this answer"):
                for r in results:
                    timestamp = f"{int(r['start_time'] // 60)}:{int(r['start_time'] % 60):02d}"
                    st.markdown(f"**[{timestamp}]** — Score: `{r['score']:.3f}`")
                    st.write(r["text"])
                    st.divider()