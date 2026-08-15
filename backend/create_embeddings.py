# ============================================================
# CREATE FAQ EMBEDDINGS
# ============================================================

# Import json so that we can read our FAQ JSON file.
import json

# Import os so that we can create folders if they don't exist.
import os

# Import pickle so that we can save our original FAQ information.
import pickle

# Import numpy for working with numerical vectors.
import numpy as np

# Import SentenceTransformer.
# This model converts text into numerical embeddings.
from sentence_transformers import SentenceTransformer

# Import FAISS.
# FAISS is used to search for similar vectors quickly.
import faiss


# ============================================================
# FILE LOCATIONS
# ============================================================

# Location of our FAQ data.
FAQ_FILE = "data/faq_data.json"

# Location where we will save the vector database.
VECTOR_FOLDER = "vector_store"

# FAISS index file.
INDEX_FILE = "vector_store/faq.index"

# Original FAQ data file.
FAQ_PICKLE_FILE = "vector_store/faq_data.pkl"


# ============================================================
# CREATE VECTOR STORE FOLDER
# ============================================================

# Create vector_store folder if it does not already exist.
os.makedirs(VECTOR_FOLDER, exist_ok=True)


# ============================================================
# LOAD FAQ DATA
# ============================================================

# Open the JSON file in read mode.
with open(FAQ_FILE, "r", encoding="utf-8") as file:

    # Convert JSON data into Python objects.
    faq_data = json.load(file)


# Print how many FAQs were loaded.
print(f"Loaded {len(faq_data)} FAQs.")


# ============================================================
# LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

# Load a lightweight and popular sentence embedding model.
#
# This model converts sentences such as:
#
# "How can I apply for internship?"
#
# into a numerical vector.
#
# Because you installed CUDA-enabled PyTorch,
# Sentence Transformers can use your NVIDIA GPU.
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# EXTRACT QUESTIONS
# ============================================================

# Create an empty list to store FAQ questions.
questions = []


# Go through every FAQ.
for faq in faq_data:

    # Add the FAQ question to our list.
    questions.append(
        faq["question"]
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("Creating embeddings...")


# Convert all FAQ questions into numerical vectors.
#
# normalize_embeddings=True makes the vectors suitable
# for cosine similarity search.
embeddings = model.encode(
    questions,
    normalize_embeddings=True,
    show_progress_bar=True
)


# Convert embeddings into float32.
# FAISS works efficiently with float32 vectors.
embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


# ============================================================
# GET VECTOR DIMENSION
# ============================================================

# Example:
# all-MiniLM-L6-v2 produces vectors with 384 dimensions.
dimension = embeddings.shape[1]


print(
    f"Embedding dimension: {dimension}"
)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

# IndexFlatIP uses inner product similarity.
#
# Since we normalized our vectors above,
# inner product becomes cosine similarity.
index = faiss.IndexFlatIP(
    dimension
)


# ============================================================
# ADD EMBEDDINGS TO FAISS
# ============================================================

# Add all FAQ embeddings to the FAISS index.
index.add(
    embeddings
)


# Print number of vectors stored.
print(
    f"Stored {index.ntotal} vectors in FAISS."
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

# Save the FAISS index to disk.
faiss.write_index(
    index,
    INDEX_FILE
)


# ============================================================
# SAVE FAQ DATA
# ============================================================

# Save the original FAQ data.
#
# FAISS only stores vectors.
# It does not store our questions and answers.
#
# Therefore we save the original FAQ data separately.
with open(
    FAQ_PICKLE_FILE,
    "wb"
) as file:

    pickle.dump(
        faq_data,
        file
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("========================================")
print("Embedding creation completed!")
print("========================================")
print(f"FAISS index: {INDEX_FILE}")
print(f"FAQ data:    {FAQ_PICKLE_FILE}")