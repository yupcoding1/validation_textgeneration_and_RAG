import numpy as np
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from rake_nltk import Rake
import fitz  # PyMuPDF for PDF handling
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

from config import GROQ_API_KEY, HF_TOKEN
# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_docs")

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDFs while keeping headings
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = []
    current_heading = None

    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if text.isupper():  # Assume uppercase text as headings
                current_heading = text
            elif current_heading:
                text_data.append(f"{current_heading}: {text}")
            else:
                text_data.append(text)

    return "\n".join(text_data)

# Function to chunk text while preserving headings
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# Function to embed and store chunks in ChromaDB
def store_chunks_in_chromadb(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = embed_model.encode(chunk).tolist()
        collection.add(ids=[f"chunk_{i}"], embeddings=[embedding], metadatas=[{"text": chunk}])

    print(f"Stored {len(chunks)} chunks from {pdf_path} in ChromaDB.")



# Initialize the Groq model
llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY,model_name="mixtral-8x7b-32768")

# -----------------------------
# Sample corpus for retrieval
# -----------------------------
CORPUS = [
    "Diabetes is a metabolic disorder characterized by high blood sugar levels.",
    "Symptoms of diabetes include frequent urination, excessive thirst, and unexplained weight loss.",
    "Treatment for diabetes involves lifestyle changes, medication, and sometimes insulin therapy.",
    "Diet and exercise are crucial in managing diabetes and preventing complications.",
    "Regular monitoring of blood sugar is important for individuals with diabetes.",
]

# -----------------------------
# Simple Retrieval Function
# -----------------------------
def retrieve_chunks1(query, corpus, top_k=3, embed_model=None):
    if embed_model is None:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    corpus_emb = embed_model.encode(corpus, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_emb, corpus_emb)[0]
    top_results = np.argsort(-cos_scores.cpu().numpy())[:top_k]
    retrieved = [corpus[idx] for idx in top_results]
    scores = [cos_scores[idx].item() for idx in top_results]
    return retrieved, scores


def retrieve_chunks(query, top_k=3,embed_model=None):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_texts = [res["text"] for res in results["metadatas"][0]]
    scores = results["distances"][0]

    return retrieved_texts, scores

# -----------------------------
# Simple Generation Function
# -----------------------------
# Generate Answer using Groq's model
def generate_answer(query, retrieved_chunks):
    input_text = "\n".join(retrieved_chunks)
    prompt_template = PromptTemplate.from_template(
        """
        Based on the following retrieved information:
        {context}
        Answer the question:
        {query}
        """
    )
    formatted_prompt = prompt_template.format(context=input_text, query=query)
    out = llm.invoke(formatted_prompt)
    return out.content

# -----------------------------
# Retrieval Evaluation Methods
# -----------------------------

# 1. Cross Encoding Evaluation
def cross_encode_evaluation(query, chunk, cross_encoder):
    # Construct an input string for the cross-encoder
    result = cross_encoder(f"Query: {query} Document: {chunk}")
    score = result[0]['score']  # higher score = better relevance
    return score

# 2. Weighted Keyword Matching Evaluation
def weighted_keyword_evaluation(query, chunk):
    # Use RAKE to extract keywords from the query
    r = Rake()
    r.extract_keywords_from_text(query)
    keywords = r.get_ranked_phrases()  # returns phrases in order of importance
    score = 0
    # Simple matching: add one point for each keyword found in the chunk
    for keyword in keywords:
        if keyword.lower() in chunk.lower():
            score += 1
    return score

# 3. Rephrasing Evaluation
def rephrasing_evaluation(query, paraphrase_model, num_paraphrases=3, top_k=3):
    paraphraser = pipeline("text2text-generation", model=paraphrase_model, token=None)
    paraphrases = []
    for _ in range(num_paraphrases):
        paraphrased = paraphraser(f"paraphrase: {query}", max_length=25, num_return_sequences=1, do_sample=True)[0]['generated_text']
        paraphrases.append(paraphrased)
    counts = {}
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    for para in paraphrases:
        retrieved, _ = retrieve_chunks(para, top_k=top_k, embed_model=embed_model)
        for chunk in retrieved:
            counts[chunk] = counts.get(chunk, 0) + 1
    return counts

# -----------------------------
# Generation Evaluation Methods
# -----------------------------

# 1. Faithfulness Check: Semantic similarity between generated answer and each retrieved chunk
def faithfulness_check(generated_output, retrieved_chunks, embed_model=None):
    if embed_model is None:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    gen_emb = embed_model.encode(generated_output, convert_to_tensor=True)
    scores = []
    for chunk in retrieved_chunks:
        chunk_emb = embed_model.encode(chunk, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(gen_emb, chunk_emb)
        scores.append(sim.item())
    return np.mean(scores)

def rag_consistency_test(query, retrieved_chunks):
    # Generate answer from retrieved chunks (RAG answer)
    rag_answer = generate_answer(query, retrieved_chunks)
    # Generate direct LLM answer without retrieval
    prompt_template = PromptTemplate.from_template(
        """
        Answer the following question without using any external context:
        {query}
        """
    )
    direct_answer = llm.invoke(prompt_template.format(query=query)).content
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    rag_emb = embed_model.encode(rag_answer, convert_to_tensor=True)
    direct_emb = embed_model.encode(direct_answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(rag_emb, direct_emb).item()
    return similarity, rag_answer, direct_answer

def llm_critique(query, generated_output, retrieved_chunks):
    prompt_template = PromptTemplate.from_template(
        """
        Query: {query}
        Retrieved Chunks: {context}
        Generated Answer: {answer}

        Does the generated answer strictly follow the retrieved chunks?.
        """
    )
    formatted_prompt = prompt_template.format(query=query, context="\n".join(retrieved_chunks), answer=generated_output)
    return llm.invoke(formatted_prompt).content

# -----------------------------
# Main Application and Evaluation
# -----------------------------

def maine(Q):

    store_chunks_in_chromadb("./temp/diabestes.pdf")
    query = Q
    eval_retrieval_methods = "cross,keyword,rephrase".split(",")
    eval_generation_methods = "faithfulness,consistency,critique".split(",")

    print("Query:", query)

    # -----------------------------
    # Step 1: RAG Retrieval and Generation
    # -----------------------------
    print("\n--- RAG Retrieval ---")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    retrieved_chunks, retrieval_scores = retrieve_chunks(query,top_k=3, embed_model=embed_model)
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Chunk {i+1} (Cosine Sim: {retrieval_scores[i]:.3f}): {chunk}")

    print("\n--- RAG Generation ---")
    generated_answer = generate_answer(query, retrieved_chunks)
    print("Generated Answer:", generated_answer)

    # -----------------------------
    # Step 2: Evaluation
    # -----------------------------
    print("\n=== Retrieval Evaluation ===")
    retrieval_evaluation_results = {}
    # Initialize cross encoder if needed
    if "cross" in eval_retrieval_methods:
        cross_encoder = pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Evaluate each retrieved chunk with selected methods
    for chunk in retrieved_chunks:
        method_scores = {}
        if "cross" in eval_retrieval_methods:
            score = cross_encode_evaluation(query, chunk, cross_encoder)
            method_scores["cross_encoding"] = score
        if "keyword" in eval_retrieval_methods:
            score = weighted_keyword_evaluation(query, chunk)
            method_scores["keyword_matching"] = score
        retrieval_evaluation_results[chunk] = method_scores

    # Rephrasing evaluation across the corpus
    if "rephrase" in eval_retrieval_methods:
        paraphrase_model = "google/byt5-small"  # Example paraphrasing model
        rephrase_counts = rephrasing_evaluation(query, paraphrase_model, num_paraphrases=3, top_k=3)
        for chunk in retrieved_chunks:
            retrieval_evaluation_results.setdefault(chunk, {})["rephrase_count"] = rephrase_counts.get(chunk, 0)

    print("Retrieval Evaluation Results:")
    for chunk, scores in retrieval_evaluation_results.items():
        print(f"Chunk: {chunk}\nScores: {scores}\n")

    print("\n=== Generation Evaluation ===")
    generation_evaluation_results = {}
    if "faithfulness" in eval_generation_methods:
        faith_score = faithfulness_check(generated_answer, retrieved_chunks, embed_model)
        generation_evaluation_results["faithfulness"] = faith_score
        print("Faithfulness (Average Semantic Similarity):", faith_score)
    if "consistency" in eval_generation_methods:
        consistency_score, rag_ans, direct_ans = rag_consistency_test(query, retrieved_chunks)
        generation_evaluation_results["consistency"] = consistency_score
        print("RAG Consistency (Similarity between RAG answer and Direct LLM answer):", consistency_score)
        print("RAG Answer:", rag_ans)
        print("Direct Answer:", direct_ans)
    if "critique" in eval_generation_methods:
        critique = llm_critique(query, generated_answer, retrieved_chunks)
        generation_evaluation_results["llm_critique"] = critique
        print("LLM Critique:", critique)
    
    return retrieval_evaluation_results, generation_evaluation_results,generated_answer



if __name__ == "__main__":
    maine()

