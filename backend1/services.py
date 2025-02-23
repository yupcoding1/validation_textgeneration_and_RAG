import language_tool_python
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from nltk.tokenize import word_tokenize


from config import GROQ_API_KEY, HF_TOKEN

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


# ✅ Load Groq models
groq_mixtral = ChatGroq(model_name="mixtral-8x7b-32768" , groq_api_key=GROQ_API_KEY)
groq_llama = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# Initialize NLP tools
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
grammar_tool = language_tool_python.LanguageTool("en-US")

prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
    You are an expert AI assistant. Provide a clear and concise answer to the following question.

    Question: {query}

    Answer:
    """
)

def web_scrape(query,search_key):
    """Scrapes Wikipedia for factual information."""
    try:
        search_url = f"https://en.wikipedia.org/wiki/{search_key}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        extracted_text = ' '.join([p.text for p in paragraphs[:5]])  # Extract first 5 paragraphs
        return extracted_text.strip()
    except Exception as e:
        return f"Error during web scraping: {e}"

def grammar_check(text):
    """Checks grammatical errors in generated content."""
    matches = grammar_tool.check(text)
    errors = [{"message": m.message, "context": m.context, "suggestions": m.replacements} for m in matches]
    return errors

def generate_answers(query):
    formatted_prompt = prompt_template.format(query=query)

    try:
        mixtral_answer = groq_mixtral.invoke(query).content
        print(mixtral_answer)
    except Exception as e:
        mixtral_answer = f"Error using Groq Mixtral: {e}"

    try:
        llama_answer = groq_llama.invoke(query).content
        print(llama_answer)
    except Exception as e:
        llama_answer = f"Error using Groq Llama: {e}"

    return {"groq_mixtral": mixtral_answer, "groq_llama": llama_answer}

def compare_texts(text1, text2):
    """Compares two texts using NLP metrics."""
    if not text1 or not text2:
        return {
            "semantic_similarity": "Comparison not possible",
            "rouge_scores": "Comparison not possible",
            "bleu_score": "Comparison not possible",
            "meteor_score": "Comparison not possible",
            "bert_score": "Comparison not possible"
        }

    # Compute Semantic Similarity
    embedding1 = semantic_model.encode(text1, convert_to_tensor=True)
    embedding2 = semantic_model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

    # Compute ROUGE Scores
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = rouge.score(text1, text2)

    # Compute BLEU Score
    bleu = sentence_bleu([word_tokenize(text1)], word_tokenize(text2), smoothing_function=SmoothingFunction().method1)

    tokenized_text1 = word_tokenize(text1)
    tokenized_text2 = word_tokenize(text2)

    # Compute METEOR Score
    meteor = meteor_score([tokenized_text1], tokenized_text2)

    # Compute BERTScore
    P, R, F1 = bert_score([text1], [text2], lang="en")
    bert_f1 = F1.item()

    return {
        "semantic_similarity": similarity,
        "rouge_scores": {k: v.fmeasure for k, v in scores.items()},
        "bleu_score": bleu,
        "meteor_score": meteor,
        "bert_score": bert_f1
    }

def compare_answers(generated, scraped):
    """Compares generated text with web-scraped content."""
    if not scraped:
        return {
            "semantic_similarity": "No reference available",
            "rouge_scores": "No reference available",
            "bleu_score": "No reference available",
            "meteor_score": "No reference available",
            "bert_score": "No reference available"
        }

    return compare_texts(generated, scraped)

def aggregate_final_answer(results):
    """Selects the best answer based on evaluation scores."""
    best_model = None
    best_score = -1
    best_answer = ""

    for model, data in results.items():
        comparison_scores = data["comparison_scores"]
        if isinstance(comparison_scores, dict) and "semantic_similarity" in comparison_scores:
            score = (
                comparison_scores["semantic_similarity"] * 0.4 +
                comparison_scores["rouge_scores"]["rougeL"] * 0.2 +
                comparison_scores["bleu_score"] * 0.2 +
                comparison_scores["meteor_score"] * 0.1 +
                comparison_scores["bert_score"] * 0.1
            )
            if score > best_score:
                best_score = score
                best_model = model
                best_answer = data["generated_answer"]

    return {"best_model": best_model, "final_answer": best_answer}

def validate(query,search_key):
    """Runs validation: Web Scraping, Model Generation, and Comparison."""

    # Step 1: Scrape factual data
    scraped_text = web_scrape(query,search_key)

    # Step 2: Generate answers from models
    generated_answers = generate_answers(query)

    # Step 3: Analyze and compare model outputs
    results = {}
    for model, generated in generated_answers.items():
        comparison = compare_answers(generated, scraped_text)
        grammar_errors = grammar_check(generated)

        results[model] = {
            "generated_answer": generated,
            "comparison_scores": comparison,
            "grammar_errors": grammar_errors
        }

    # Step 4: Compare Model 1 vs. Model 2
    model_comparison = compare_texts(generated_answers["groq_mixtral"], generated_answers["groq_llama"])

    # Step 5: Determine the final answer
    final_answer = aggregate_final_answer(results)

    return {
        "web_scraped_content": scraped_text,
        "validation_results": results,
        "model_comparison": model_comparison,
        "final_answer": final_answer
    }

if __name__ == "__main__":
    query = "What is quantum computing?"
    results = validate(query,"Quantum_computing")

    print("\nWeb-Scraped Content:\n", results["web_scraped_content"])

    print("\nValidation Results:")
    for model, data in results["validation_results"].items():
        print(f"\nModel: {model}")
        print("Generated Answer:\n", data["generated_answer"])
        print("Comparison Scores:\n", data["comparison_scores"])
        print("Grammar Errors:\n", data["grammar_errors"])

    print("\nModel-to-Model Comparison (Mistral vs. Llama-2):")
    print(results["model_comparison"])

    print("\nFinal Answer:")
    print(f"Best Model: {results['final_answer']['best_model']}")
    print(f"Final Answer: {results['final_answer']['final_answer']}")


