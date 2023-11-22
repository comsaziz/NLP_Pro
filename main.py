
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rank_documents(documents, dictionary, query):
    # Convert documents to lowercase
    documents_lower = [doc.lower() for doc in documents]

    # Convert dictionary to lowercase
    dictionary_lower = [word.lower() for word in dictionary]

    # Calculate TF values
    tf_matrix = np.zeros((len(documents_lower), len(dictionary_lower)))
    for i, doc in enumerate(documents_lower):
        for j, word in enumerate(dictionary_lower):
            tf_matrix[i, j] = doc.count(word) / len(doc.split())

    # Calculate IDF values
    total_documents = len(documents_lower)
    idf_vector = np.log2(total_documents / np.count_nonzero(tf_matrix, axis=0))

    # Calculate TF-IDF values
    tfidf_matrix = tf_matrix * idf_vector

    # Normalize TF-IDF values
    tfidf_norm = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    tfidf_matrix /= tfidf_norm

    # Convert the query to lowercase
    query_lower = query.lower()

    # Calculate TF values for the query
    query_tf = np.array([(query_lower.count(word) / len(query_lower.split())) for word in dictionary_lower])

    # Calculate IDF values for the query
    query_idf = np.log2(total_documents / np.count_nonzero(tf_matrix, axis=0))

    # Calculate TF-IDF values for the query
    query_tfidf = query_tf * query_idf

    # Normalize TF-IDF values for the query
    query_tfidf /= np.linalg.norm(query_tfidf)

    # Calculate cosine similarity between the query and each document
    cosine_similarities = cosine_similarity(tfidf_matrix, [query_tfidf])

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(data={'Document': documents, 'Cosine Similarity': cosine_similarities.flatten()})

    # ---------------------- Ranking Starts Here ----------------------
    # Rank the documents based on cosine similarity
    results_df['Rank'] = results_df['Cosine Similarity'].rank(ascending=False)

    return results_df.sort_values(by='Rank')

# Given documents and dictionary
documents = [
    "Vaccination Application",
    "Covid Vaccination Center",
    "Health of Pilgrims",
    "Certificate of Vaccination"
]

dictionary = ["Application", "Vaccination", "Covid", "Pilgrims", "Health", "Certificate", "Center"]

# Given query
query = "Covid Vaccination"

# Call the function and display the results
ranked_documents = rank_documents(documents, dictionary, query)
print("\nRanked Documents for the Query 'Covid Vaccination':")
print(ranked_documents)