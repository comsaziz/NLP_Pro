from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# # Define the documents
# documents = [
#     "Data Base System Concepts",
#     "Introduction to Algorithms",
#     "Computer Geometry Application",
#     "Data Structure and Algorithm Analysis on Data"
# ]

# # Define the dictionary (7 words)
# dictionary = ["Data", "System", "Algorithm", "Computer", "Geometry", "Structure", "Analysis"]

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

# Now use documents_lower and dictionary_lower in your TF-IDF calculations

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

#---------------------------------------------------------------
tfidf_norm = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
tfidf_matrix /= tfidf_norm

# Display the normalized TF-IDF values as a DataFrame
tfidf_df = pd.DataFrame(data=tfidf_matrix, columns=dictionary)
print("\nNormalized TF-IDF values:")
print(tfidf_df)