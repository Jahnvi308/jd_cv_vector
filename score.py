import faiss
import numpy as np

# Load the first vector database index
index1 = faiss.read_index("C:/jahnvi/python/jdcv/vectoro_db.index")

# Load the second vector database index
index2 = faiss.read_index("C:/jahnvi/python/jdcv/vector1_db.index")

# Retrieve the word vectors from the first database index
# Retrieve the vectors separately if using IndexFlat
vectors1 = np.array([index1.reconstruct(i) for i in range(index1.ntotal)])

# Retrieve the word vectors from the second database index
# Retrieve the vectors separately if using IndexFlat
vectors2 = np.array([index2.reconstruct(i) for i in range(index2.ntotal)])

# Calculate the cosine similarity between each vector from the second database
# and each vector from the first database
similarities = []
for vector2 in vectors2:
    similarity_scores = []
    for vector1 in vectors1:
        similarity_scores.append(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    average_similarity = sum(similarity_scores) / len(similarity_scores)
    similarities.append(average_similarity)

# Calculate the overall similarity score
overall_similarity = sum(similarities) / len(similarities)
print(overall_similarity)
