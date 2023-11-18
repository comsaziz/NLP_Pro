def compute_tf(document, vocabulary):
    tf_values = {}
    for word in vocabulary:
        tf_values[word] = document.count(word)
    return tf_values


   