import chromadb
from sentence_transformers import SentenceTransformer
import json

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="semanticSentences")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def generate_embedding(text):
    return(model.encode(text))


sentences = [
    "Artificial intelligence is transforming the way businesses operate.",
    "The capital of France is Paris.",
    "Customer support can be automated using chatbots and natural language processing.",
    "The quick brown fox jumps over the lazy dog.",
    "Renewable energy sources are key to combating climate change.",
    "Vector databases store embeddings for efficient similarity search.",
    "Machine learning models improve as they are exposed to more data.",
    "Python is one of the most popular programming languages in data science.",
    "The Mona Lisa is one of the most famous paintings in the world.",
    "Quantum computing could revolutionize cryptography and optimization problems.",
    "Regular exercise and a balanced diet contribute to a healthy lifestyle.",
    "OpenAI develops advanced AI models for text, image, and code generation.",
    "Space exploration helps humanity understand more about the universe.",
    "The weather today is sunny with a chance of light rain in the evening.",
    "Cloud computing allows users to access resources and services over the internet.",
    "Cybersecurity is essential to protect sensitive data from unauthorized access.",
    "Dogs are known for their loyalty and companionship.",
    "The global economy is influenced by trade, technology, and politics.",
    "Music has the power to evoke emotions and connect people across cultures.",
    "Learning a new language can expand career opportunities and cultural awareness."
]

sentence_vectors = [generate_embedding(sentence) for sentence in sentences]
sentence_ids = [f"id_{i+1}" for i in range(len(sentences))]


collection.add(
    ids=sentence_ids,
    documents=sentences, 
    embeddings=sentence_vectors
)


query = "Ali is the ariabian batman"
embedded_query = [generate_embedding(query)]
results = collection.query(
    query_embeddings=embedded_query,
    n_results=5
)

print("Top 5 similar sentences:")
for doc in results['documents'][0]:
    print("-", doc)

items = collection.get(
    include=["metadatas", "documents", "embeddings"]
)

items["embeddings"] = [embedding.tolist() for embedding in items["embeddings"]]

with open("sentence_vectors.json", "w") as f:
    json.dump(items, f, indent=4)

