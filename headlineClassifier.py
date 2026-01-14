from sentence_transformers import SentenceTransformer # type: ignore
from transformers import pipeline # type: ignore
import numpy as np # type: ignore

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

categories = {"Tech":"Technology hardware, software, and services, Apple, Facebook, Nvidia, Netflix, Microsoft, AI, OpenAI",
    "Finance":"Banks, investment banks, brokers, financial exhanges",
    "Commodities":"Oil, gas, petroleum, refining, pipelines, chemicals, metals, crops, wheat, barley, oats, corn, gold, silver, copper, bauxite",
    "Industrials":"Aerospace, defense, machinery, construction, transportation",
    "Healthcare":"Pharmaceuticals, biotechnology, medical devices, health insurance, hospitals",
    "Retail":"Retail, cars, leisure, hotels, casinos, e-commerce, clothes, food & beverage, supermarkets, household",
    "Real Estate":"REITs, commercial and residential real estate, interest rates",
    "Bonds":"US Treasury notes, Bonds, National debt, interest rates",
    "Derivatives":"Futures, Options, Swaps, Forwards, Credit Default Swaps, Interest Rate Swaps, Currency Swaps, Equity Options, Commodity Futures, Total Return Swaps",
    "Communication":"Telecom, media, broadcasting, television, newspapers"
 }

category_names = list(categories.keys())
category_texts = list(categories.values())

category_embeddings = model.encode(
    category_texts,
    normalize_embeddings=True
)

def classify_text(text, threshold=0.05):
    """
    Classify text into the closest category using cosine similarity.
    Returns category.
    """
    text_embedding = model.encode(
        [text],
        normalize_embeddings=True
    )[0]

    # Cosine similarity via dot product (normalized vectors)
    similarities = np.dot(category_embeddings, text_embedding)

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score < threshold:
        return "other"

    return category_names[best_idx]

def classify_sentiment(text):
    """
    Classify text into the closest sentiment using cosine similarity.
    Returns category.
    """
    result = sentiment_pipeline(text, truncation=True)[0]

    return result["label"].lower()