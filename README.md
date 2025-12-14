# Text Summarizer & Topic Classifier

A Streamlit-based NLP application that summarizes long text or PDF documents and classifies them into topic categories using Hugging Face transformer models.
The summary length dynamically adjusts based on the size of the input text, ensuring concise yet meaningful output.

---

## Features

- Abstractive text summarization using transformer models
- Topic classification with confidence score
- Supports plain text input and PDF upload
- Dynamic summary length proportional to input size
- Word count and character count display
- Clean, tab-based Streamlit UI
- CPU-optimized with cached models

---

## Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- PyPDF

---

## Models Used

| Task | Model |
|-----|-------|
| Summarization | facebook/bart-large-cnn |
| Topic Classification | cardiffnlp/tweet-topic-21-multi |

---

## Project Structure

text-summarizer-classifier/
│
├── app.py
├── requirements.txt
└── README.md

---

## Installation

### Clone the repository
git clone https://github.com/your-username/text-summarizer-classifier.git
cd text-summarizer-classifier

### Create virtual environment
python -m venv venv

Windows:
venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

---

## Run the Application

streamlit run app.py

---
<img width="1919" height="917" alt="Screenshot 2025-12-14 211645" src="https://github.com/user-attachments/assets/2b00de2b-f4d6-4b03-99b8-d192f20a12a2" />

## Usage

1. Paste text or upload a PDF document
2. View word count and character count
3. Click Analyze Text
4. Navigate between:
   - Summary tab for generated summary
   - Topic Classification tab for predicted topic and confidence score

---

<img width="1919" height="915" alt="Screenshot 2025-12-14 211855" src="https://github.com/user-attachments/assets/c58c1dd4-c2ae-40b2-8714-bf45643ba9ab" />


<img width="1919" height="915" alt="Screenshot 2025-12-14 212053" src="https://github.com/user-attachments/assets/6c1c261f-43d7-444c-af39-f66f62fa2d6b" />



## Dynamic Summary Logic

The application adjusts summary length automatically:

- Small inputs → short summaries
- Medium inputs → moderate summaries
- Large inputs → more detailed summaries

This ensures readability and relevance across document sizes.

---

## Supported Inputs

- Plain text (articles, blogs, reports)
- PDF documents (multi-page supported)

---

## Future Improvements

- DOCX support
- Download summary as PDF
- Faster summarization models
- Multi-label topic classification
- REST API using FastAPI

---

