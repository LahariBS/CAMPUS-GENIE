# ============================================================
# CREATE FAQ EMBEDDINGS
# ============================================================

import json
import os
import pickle

import numpy as np
import faiss

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# FILE LOCATIONS
# ============================================================

FAQ_FILE = "data/faq_data.json"

VECTOR_FOLDER = "vector_store"

INDEX_FILE = "vector_store/faq.index"

FAQ_PICKLE_FILE = "vector_store/faq_data.pkl"

VECTORIZER_FILE = "vector_store/tfidf_vectorizer.pkl"


# ============================================================
# CREATE VECTOR STORE FOLDER
# ============================================================

os.makedirs(
    VECTOR_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD FAQ DATA
# ============================================================

with open(
    FAQ_FILE,
    "r",
    encoding="utf-8"
) as file:

    faq_data = json.load(file)


print(
    f"Loaded {len(faq_data)} FAQs."
)


# ============================================================
# EXTRACT QUESTIONS
# ============================================================

questions = []

for faq in faq_data:

    questions.append(
        faq["question"]
    )


# ============================================================
# CREATE TF-IDF VECTORIZER
# ============================================================

print("Creating FAQ embeddings...")


vectorizer = TfidfVectorizer(
    stop_words="english"
)


# Convert FAQ questions into numerical vectors.
embeddings = vectorizer.fit_transform(
    questions
)


# Convert sparse matrix into dense float32.
embeddings = embeddings.toarray().astype(
    "float32"
)


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

# Normalize vectors so inner product behaves
# similarly to cosine similarity.

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

norms[norms == 0] = 1

embeddings = embeddings / norms


# ============================================================
# GET VECTOR DIMENSION
# ============================================================

dimension = embeddings.shape[1]

print(
    f"Embedding dimension: {dimension}"
)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

index = faiss.IndexFlatIP(
    dimension
)


# ============================================================
# ADD EMBEDDINGS TO FAISS
# ============================================================

index.add(
    embeddings
)


print(
    f"Stored {index.ntotal} vectors in FAISS."
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

faiss.write_index(
    index,
    INDEX_FILE
)


# ============================================================
# SAVE FAQ DATA
# ============================================================

with open(
    FAQ_PICKLE_FILE,
    "wb"
) as file:

    pickle.dump(
        faq_data,
        file
    )


# ============================================================
# SAVE TF-IDF VECTORIZER
# ============================================================

with open(
    VECTORIZER_FILE,
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("========================================")
print("Embedding creation completed!")
print("========================================")

print(
    f"FAISS index:       {INDEX_FILE}"
)

print(
    f"FAQ data:          {FAQ_PICKLE_FILE}"
)

print(
    f"TF-IDF vectorizer: {VECTORIZER_FILE}"
)