import faiss
import numpy as np
# Load the first vector database index
index1 = faiss.read_index("C:/jahnvi/python/jdcv/vector_db.index")

# Load the second vector database index
index2 = faiss.read_index("C:/jahnvi/python/jdcv/vectoro_db.index")

# Retrieve the word vectors from the first database index
# Retrieve the vectors separately if using IndexFlat
vectors1 = np.array([index1.reconstruct(i) for i in range(index1.ntotal)])

# Retrieve the word vectors from the second database index
# Retrieve the vectors separately if using IndexFlat
vectors2 = np.array([index2.reconstruct(i) for i in range(index2.ntotal)])

distances = []
for vector2 in vectors2:
    distance_scores = []
    for vector1 in vectors1:
        distance_scores.append(faiss.vector_L2sqr(vector1, vector2))
    average_distance = sum(distance_scores) / len(distance_scores)
    distances.append(average_distance)

overall_similarity = 1 / (sum(distances) / len(distances))
print(overall_similarity)