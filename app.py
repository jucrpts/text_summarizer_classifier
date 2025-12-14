import os
import streamlit as st
from transformers import pipeline
from pypdf import PdfReader

# ---------------- Stability Fix ----------------
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Text Summarizer & Topic Classifier",
    layout="centered"
)

st.title("📝 Text Summarizer & Topic Classifier")
st.caption(
    "Summarize long documents and classify them into topics using NLP models."
)

# ---------------- Load Models ----------------

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1
    )

@st.cache_resource
def load_classifier():
    return pipeline(
        "text-classification",
        model="cardiffnlp/tweet-topic-21-multi",
        top_k=1,
        device=-1
    )

summarizer = load_summarizer()
classifier = load_classifier()

# ---------------- Helper Functions ----------------

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def safe_truncate(text, max_chars=3500):
    return text[:max_chars] if len(text) > max_chars else text

def dynamic_summary_length(word_count):
    """
    Generate proportional summary lengths
    """
    if word_count < 300:
        return 40, 100
    elif word_count < 800:
        return 60, 150
    else:
        return 80, 200

# ---------------- Input Section ----------------

st.subheader("📥 Input")

input_mode = st.radio(
    "Choose input type:",
    ["Paste Text", "Upload PDF"],
    horizontal=True
)

text_input = ""

if input_mode == "Paste Text":
    text_input = st.text_area(
        "Paste your text",
        height=250,
        placeholder="Paste article, report, or document text here..."
    )

else:
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )
    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            text_input = extract_text_from_pdf(uploaded_file)

# ---------------- Stats ----------------

if text_input.strip():
    word_count = len(text_input.split())
    char_count = len(text_input)

    col1, col2 = st.columns(2)
    col1.metric("Word Count", word_count)
    col2.metric("Character Count", char_count)

# ---------------- Action ----------------

st.divider()

if st.button("🚀 Analyze Text"):
    if text_input.strip() == "":
        st.warning("Please provide some text or upload a PDF.")
    elif len(text_input.split()) < 50:
        st.warning("Please provide at least 50 words.")
    else:
        with st.spinner("Summarizing and classifying..."):
            try:
                truncated_text = safe_truncate(text_input)

                min_len, max_len = dynamic_summary_length(
                    len(truncated_text.split())
                )

                summary_output = summarizer(
                    truncated_text,
                    min_length=min_len,
                    max_length=max_len,
                    do_sample=False,
                    truncation=True
                )

                summary_text = summary_output[0]["summary_text"]

                classification_output = classifier(summary_text)
                top_prediction = classification_output[0][0]

                # ---------------- Results Tabs ----------------

                tab1, tab2 = st.tabs(["📄 Summary", "🏷 Topic Classification"])

                with tab1:
                    st.subheader("Generated Summary")
                    st.write(summary_text)
                    st.caption(
                        f"Summary length: {len(summary_text.split())} words"
                    )

                with tab2:
                    st.subheader("Predicted Topic")
                    st.markdown(f"### {top_prediction['label']}")
                    st.progress(min(top_prediction["score"], 1.0))
                    st.caption(
                        f"Confidence Score: {top_prediction['score']:.2f}"
                    )

            except Exception as e:
                st.error("An error occurred during processing.")
                st.code(str(e))
