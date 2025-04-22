import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline
from newspaper import Article
import nltk
import time

# Download NLTK data if not already available
nltk.download('punkt')
nltk.download('stopwords')

# ----------- Summarization Functions -----------

def sumy_summarizer(text, summarizer_type='lsa', sentences_count=5):
    """Extractive summarization using Sumy library"""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    if summarizer_type == 'lsa':
        summarizer = LsaSummarizer()
    elif summarizer_type == 'lex':
        summarizer = LexRankSummarizer()
    elif summarizer_type == 'luhn':
        summarizer = LuhnSummarizer()
    elif summarizer_type == 'textrank':
        summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join([str(sentence) for sentence in summary])

def abstractive_summarizer(text, max_length=130, min_length=30):
    """Abstractive summarization using BART model"""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=max_length, min_length=min_length)[0]['summary_text']

def get_article_text(url):
    """Extract text from news article URL"""
    article = Article(url)
    article.download()
    article.parse()
    return article.text

# ----------- Streamlit App -----------

def main():
    st.set_page_config(page_title="Article Summarizer", layout="wide")

    st.title("📝 Article Summarization Tool")
    st.markdown("""
    Summarize lengthy articles using NLP techniques.
    Paste text directly or provide a URL.
    """)

    # Input options
    input_method = st.radio("Input method:", ("Text", "URL"), horizontal=True)
    text = ""

    if input_method == "Text":
        text = st.text_area("Paste your article text here:", height=200)
    else:
        url = st.text_input("Enter article URL:")
        if url:
            with st.spinner("Fetching article..."):
                try:
                    text = get_article_text(url)
                    st.text_area("Extracted Text:", text, height=200)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if text:
        # Summarization options
        st.subheader("Summarization Options")
        col1, col2 = st.columns(2)

        with col1:
            summary_type = st.selectbox(
                "Summary type:",
                ("Extractive (LSA)", "Extractive (LexRank)",
                 "Extractive (Luhn)", "Extractive (TextRank)",
                 "Abstractive (BART)")
            )

        with col2:
            length = st.slider(
                "Summary length:",
                min_value=1, max_value=10, value=5
            )

        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                try:
                    start_time = time.time()

                    if "Extractive" in summary_type:
                        method = summary_type.split("(")[1].split(")")[0].lower()
                        summary = sumy_summarizer(text, method, length)
                        st.subheader("📋 Extractive Summary")
                    else:
                        summary = abstractive_summarizer(text, max_length=length * 30)
                        st.subheader("✨ Abstractive Summary")

                    st.write(summary)

                    # Stats
                    orig_words = len(word_tokenize(text))
                    summ_words = len(word_tokenize(summary))
                    ratio = (orig_words - summ_words) / orig_words * 100

                    st.success(f"""
                    **Summary Statistics**
                    - Original: {orig_words} words
                    - Summary: {summ_words} words
                    - Reduced by: {ratio:.1f}%
                    - Time taken: {time.time() - start_time:.2f}s
                    """)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
