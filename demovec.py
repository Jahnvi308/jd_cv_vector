import PyPDF2
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models import Word2Vec
nltk.download('wordnet')
import numpy as np
import faiss


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove line breaks and extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove bullet points and symbols
    text = re.sub(r'•|●|∙|-|\*|·', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stopwords_list = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stopwords_list]
    

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a single string
    processed_text = " ".join(tokens)

    return processed_text

# Open the PDF file
with open("C:/jahnvi/python/jdcv/Pairs/1/job_descriptions/JPC5663_Java Developer.pdf", "rb") as file:
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(file)

    # Extract text from each page
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

# Preprocess the extracted text
processed_text = preprocess_text(text)

# Print the preprocessed text
# print(processed_text)


# Train Word2Vec model
model = Word2Vec([processed_text], min_count=1)

# Get the vector representation of a word
# vector = model.wv["business"]
# Create a dictionary to store the word vectors
word_to_vec = {word: model.wv.get_vector(word) for word in model.wv.index_to_key}

# Convert word vectors to a NumPy array
vectors = np.array(list(word_to_vec.values()))

# Initialize the FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])

# Add vectors to the index
index.add(vectors)

# Save the index
faiss.write_index(index, "vectoro_db.index")

# Print the Word2Vec model
print(index)