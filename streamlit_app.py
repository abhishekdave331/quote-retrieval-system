# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

# Load cleaned dataset
df = pd.read_csv("cleaned_quotes.csv")

# Load models
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_model = pipeline("text2text-generation", model="google/flan-t5-base")
    return embed_model, gen_model

# Load model globally, outside of caching
model, generator = load_models()

# Create FAISS index — now ONLY pass plain hashable types like strings/lists
@st.cache_resource
def create_faiss_index(quotes):
    embeddings = model.encode(quotes, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Call function without passing the model
index, quote_embeddings = create_faiss_index(df['quote'].tolist())

# Retrieval function
def retrieve_top_k(query, k=5):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = distances[0].round(2)
    return results

# RAG function
def rag_pipeline(query, k=5):
    retrieved_df = retrieve_top_k(query, k)
    context = "\n".join(retrieved_df['quote'].tolist())
    prompt = f"Given the following quotes:\n{context}\n\nAnswer the query: {query}"
    response = generator(prompt, max_length=100, do_sample=False)[0]["generated_text"]
    return response, retrieved_df

# Streamlit UI
st.set_page_config(page_title="Semantic Quote Search", layout="wide")
st.markdown(
    "<h1 style='text-align: left; color: #4B8BBE;'>✨ Semantic Quote Explorer ✨</h1>",
    unsafe_allow_html=True
)

with st.expander("📖 What does this app do?", expanded=True):
    st.markdown(
        "This app lets you search meaningful quotes using natural language. "
        "It uses a semantic retrieval pipeline powered by Sentence Transformers, FAISS, and Flan-T5."
    )

with st.container():
    st.subheader("🔎 Search")
    query = st.text_input("Type your query:", placeholder="e.g. quotes about loneliness by authors who overcame adversity")
    top_k = st.slider("Number of Quotes to Retrieve", 1, 10, 5)


if st.button("Search") and query:
    with st.spinner("Retrieving and generating summary..."):
        summary, quotes_df = rag_pipeline(query, k=top_k)

    st.subheader("📜 Generated Summary")
    st.success(summary)

    st.subheader("🔍 Retrieved Quotes")
    st.dataframe(quotes_df[['quote', 'author', 'tags', 'similarity_score']])

for idx, row in quotes_df.iterrows():
    with st.expander(f"💬 Quote {idx+1} (Score: {np.round(row['similarity_score'], 2)})"):
        st.markdown(f"**Quote**: {row['quote']}")
        st.markdown(f"**Author**: `{row['author']}`")
        st.markdown(f"**Tags**: {', '.join(eval(row['tags'])) if isinstance(row['tags'], str) else row['tags']}")

st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Made with ❤️ by Abhishek | "
    "<a href='https://github.com/abhishekdave331' style='color: gray;' target='_blank'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
