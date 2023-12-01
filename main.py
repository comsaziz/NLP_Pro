from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Given documents
documents = [
    "Vaccination Application",
    "Covid Vaccination Center",
    "Health of Pilgrims",
    "Certificate of Vaccination"
]

# Given dictionary
dictionary = ["Application", "Vaccination", "Covid", "Pilgrims", "Health", "Certificate", "Center"]

# Convert documents to lowercase
documents_lower = [doc.lower() for doc in documents]

# Convert dictionary to lowercase
dictionary_lower = [word.lower() for word in dictionary]

# Calculate TF values
tf_matrix = np.zeros((len(documents_lower), len(dictionary_lower)))
for i, doc in enumerate(documents_lower):
    for j, word in enumerate(dictionary_lower):
        tf_matrix[i, j] = doc.count(word) / len(doc.split())

# Display the TF values as a DataFrame
tf_df = pd.DataFrame(data=tf_matrix, columns=dictionary_lower)
print("TF values:")
print(tf_df)

# Calculate IDF values
total_documents = len(documents_lower)
idf_vector = np.log2(total_documents / np.count_nonzero(tf_matrix, axis=0))

# Display the IDF values as a DataFrame
idf_df = pd.DataFrame(data={'IDF': idf_vector}, index=dictionary_lower)
print("\nIDF values:")
print(idf_df)

# Calculate TF-IDF values
tfidf_matrix = tf_matrix * idf_vector

# Display the TF-IDF values as a DataFrame
tfidf_df = pd.DataFrame(data=tfidf_matrix, columns=dictionary_lower)
print("\nTF-IDF values:")
print(tfidf_df)




#------Start of Nomalized code section






# Normalize TF-IDF values
tfidf_norm = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
tfidf_matrix /= tfidf_norm







# Display the normalized TF-IDF values as a DataFrame
tfidf_df_normalized = pd.DataFrame(data=tfidf_matrix, columns=dictionary_lower)
print("\nNormalized TF-IDF values:")
print(tfidf_df_normalized)




#------Start of Ranking code section
# Query
query = "Covid Vaccination"

# Convert the query to lowercase
query_lower = query.lower()

# Calculate TF values for the query
query_tf = np.zeros((1, len(dictionary_lower)))
for j, word in enumerate(dictionary_lower):
    query_tf[0, j] = query_lower.count(word) / len(query_lower.split())

# Calculate IDF values for the query
query_idf = np.log2(total_documents / np.count_nonzero(tf_matrix, axis=0))

# Calculate TF-IDF values for the query
query_tfidf = query_tf * query_idf

# Normalize the TF-IDF values for the query
query_tfidf_norm = np.linalg.norm(query_tfidf)

# Calculate Cosine Similarity scores
cosine_sim_scores = cosine_similarity(tfidf_matrix, query_tfidf)[:, 0]

# Create a DataFrame to display the results
result_df = pd.DataFrame(data={'Document': documents, 'Cosine Similarity Score': cosine_sim_scores})

# Rank the documents based on the scores
result_df['Rank'] = result_df['Cosine Similarity Score'].rank(ascending=False)

# Display the ranked results
print("\nRanked Results for the Query 'Covid Vaccination':")
print(result_df.sort_values(by='Rank'))