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

model, generator = load_models()

# Create FAISS index
@st.cache_resource
def create_faiss_index(quotes, model):
    embeddings = model.encode(quotes, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

index, quote_embeddings = create_faiss_index(df['quote'].tolist(), model)

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

st.title("📚 Semantic Quote Explorer")
st.write("Search through quotes using natural language. Powered by Sentence Transformers, FAISS, and RAG with Flan-T5.")

query = st.text_input("🔎 Enter your quote-related query:", placeholder="e.g. quotes about failure by female scientists")

top_k = st.slider("Top K Quotes to Retrieve", 1, 10, 5)

if st.button("Search") and query:
    with st.spinner("Retrieving and generating summary..."):
        summary, quotes_df = rag_pipeline(query, k=top_k)

    st.subheader("📜 Generated Summary")
    st.success(summary)

    st.subheader("🔍 Retrieved Quotes")
    st.dataframe(quotes_df[['quote', 'author', 'tags', 'similarity_score']])

