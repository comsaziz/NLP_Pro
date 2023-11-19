from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Define the documents
documents = [
    "Data Base System Concepts",
    "Introduction to Algorithms",
    "Computer Geometry Application",
    "Data Structure and Algorithm Analysis on Data"
]

# Define the dictionary (7 words)
dictionary = ["Data", "System", "Algorithm", "Computer", "Geometry", "Structure", "Analysis"]

# Calculate TF values
tf_matrix = np.zeros((len(documents), len(dictionary)))
for i, doc in enumerate(documents):
    for j, word in enumerate(dictionary):
        tf_matrix[i, j] = doc.count(word) / len(doc.split())

# Display the TF values as a DataFrame
tf_df = pd.DataFrame(data=tf_matrix, columns=dictionary)
print("TF values:")
print(tf_df)

# Calculate IDF values
total_documents = len(documents)
idf_vector = np.log2(total_documents / np.count_nonzero(tf_matrix, axis=0))

# Display the IDF values as a DataFrame
idf_df = pd.DataFrame(data={'IDF': idf_vector}, index=dictionary)
print("\nIDF values:")
print(idf_df)

# Calculate TF-IDF values
tfidf_matrix = tf_matrix * idf_vector

# Display the TF-IDF values as a DataFrame
tfidf_df = pd.DataFrame(data=tfidf_matrix, columns=dictionary)
print("\nTF-IDF values:")
print(tfidf_df)