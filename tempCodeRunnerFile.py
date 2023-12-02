from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Data Base System Concepts",
    "Introduction to Algorithms",
    "Computer Geometry Application",
    "Data Structure and Algorithm Analysis on Data"
]

dictionary = ["data", "system", "algorithm", "computer", "geometry", "structure", "analysis"]

documents_lower = [doc.lower() for doc in documents]
dictionary_lower = [word.lower() for word in dictionary]

tf_matrix = np.zeros((len(documents_lower), len(dictionary_lower)))
for i, doc in enumerate(documents_lower):
    for j, word in enumerate(dictionary_lower):
        tf_matrix[i, j] = doc.count(word) / len(doc.split())

tf_df = pd.DataFrame(data=tf_matrix, columns=dictionary_lower)
print("TF values:")
print(tf_df)

total_documents = len(documents_lower)
idf_vector = np.log2(total_documents / np.count_nonzero(tf_matrix, axis=0))

idf_df = pd.DataFrame(data={'IDF': idf_vector}, index=dictionary_lower)
print("\nIDF values:")
print(idf_df)

tfidf_matrix = tf_matrix * idf_vector

tfidf_df = pd.DataFrame(data=tfidf_matrix, columns=dictionary_lower)
print("\nTF-IDF values:")
print(tfidf_df)

tfidf_norm = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
tfidf_matrix /= tfidf_norm

tfidf_df_normalized = pd.DataFrame(data=tfidf_matrix, columns=dictionary_lower)
print("\nNormalized TF-IDF values:")
print(tfidf_df_normalized)

query = "Data Analysis"
query_lower = query.lower()

query_tf = np.zeros((1, len(dictionary_lower)))
for j, word in enumerate(dictionary_lower):
    query_tf[0, j] = query_lower.count(word) / len(query_lower.split())

query_idf = np.log2(total_documents / np.count_nonzero(tf_matrix, axis=0))

query_tfidf = query_tf * query_idf

query_tfidf_norm = np.linalg.norm(query_tfidf)

cosine_sim_scores = cosine_similarity(tfidf_matrix, query_tfidf)[:, 0]

result_df = pd.DataFrame(data={'Document': documents, 'Cosine Similarity Score': cosine_sim_scores})

result_df['Rank'] = result_df['Cosine Similarity Score'].rank(ascending=False)

print("\nRanked Results for the Query 'Data Analysis':")
print(result_df.sort_values(by='Rank'))