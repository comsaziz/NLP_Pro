from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Define the documents
documents = [
    "Vaccination Applications",
    "Covid Vaccination Center",
    "Health of Pilgrims",
    "Certificate of Vaccination"
]

# Define the dictionary (7 words)
dictionary = ["Application", "Vaccination", "Covid", "Pilgrims", "Health", "Certificate", "Center"]

# Calculate TF values
tf_matrix = np.zeros((len(documents), len(dictionary)))
for i, doc in enumerate(documents):
    for j, word in enumerate(dictionary):
        tf_matrix[i, j] = doc.count(word) / len(doc.split())

# Display the TF values as a DataFrame
TF = pd.DataFrame(data=tf_matrix, columns=dictionary)
print("TF values:")
print(TF)

# Calculate IDF values
total_documents = len(documents)
df_vector = np.count_nonzero(tf_matrix, axis=0)
idf_vector = np.log(total_documents / (1 + df_vector))

# Display the corrected IDF values
idf_values_dict = dict(zip(dictionary, idf_vector))
for word, idf_value in idf_values_dict.items():
    print(f"IDF({word}) ≈ {idf_value:.7f}")

# Calculate TF-IDF values
tfidf_matrix = tf_matrix * idf_vector

# Display the TF-IDF values as a DataFrame
tfidf_df = pd.DataFrame(data=tfidf_matrix, columns=dictionary)
print("\nTF-IDF values:")
print(tfidf_df)