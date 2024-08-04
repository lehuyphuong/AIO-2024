import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


vi_data_df = pd.read_csv(
    "D:/2024/AIO2024/GIT/AIO-2024/Module_2/Week_4/vi_text_retrieval.csv")


def pearson_correlation(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2)
                          * np.sum((y - mean_y) ** 2))
    return numerator / denominator if denominator != 0 else 0


def tfidf_search(question, tfidf_vectorizer, top_d):
   # Lowercasing before encoding
    question = question.lower()

    # Encoding the query using the TF-IDF vectorizer
    query_embedded = tfidf_vectorizer.fit_transform([question])

    # Computing the cosine similarity between the query and all document embeddings
    cosine_scores = cosine_similarity(query_embedded, tfidf_vectorizer.transform(
        tfidf_vectorizer.get_feature_names_out())).flatten()

    # Get top k cosine scores and their indices
    results = []
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            'id': idx,
            'cosine_score': cosine_scores[idx]
        }
        results.append(doc_score)

    return results


def corr_search(question, tfidf_vectorizer, top_d=5):
    # Lowercasing before encoding
    question = question.lower()

    # Encoding the query using the TF-IDF vectorizer
    query_embedded = tfidf_vectorizer.fit_transform(
        [question]).toarray().flatten()

    # Computing the correlation scores between the query and all document embeddings
    corr_scores = []
    for doc_embedding in tfidf_vectorizer.transform(tfidf_vectorizer.get_feature_names_out()).toarray():
        corr_score = pearson_correlation(
            query_embedded, doc_embedding.flatten())
        corr_scores.append(corr_score)
    corr_scores = np.array(corr_scores)

    # Get top k correlation scores and their indices
    results = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
        doc = {
            'id': idx,
            'corr_score': corr_scores[idx]
        }
        results.append(doc)

    return results


if __name__ == "__main__":

    context = vi_data_df['text']
    context = [doc.lower() for doc in context]

    tfidf_vectorizer = TfidfVectorizer()

    context_embedded = tfidf_vectorizer.fit_transform(context)
    print(context_embedded.toarray()[7][0])

    tfidf_vectorizer_1 = TfidfVectorizer()
    question = vi_data_df.iloc[0]['question']
    results = tfidf_search(question, tfidf_vectorizer_1, top_d=5)
    print(results[0]['cosine_score'])

    tfidf_vectorizer_2 = TfidfVectorizer()
    question = vi_data_df.iloc[0]['question']
    results = corr_search(question, tfidf_vectorizer_2, top_d=5)
    print(results[1]['corr_score'])
