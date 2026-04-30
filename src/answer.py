import pandas as pd
import numpy as np
import openai
import tiktoken
from typing import List, Dict
import os
import json
import boto3
import io


os.environ["TOKENIZERS_PARALLELISM"] = "false"
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
openai.api_key = os.environ.get("OPENAI_API_KEY")


session = boto3.Session()

s3 = session.client('s3')
bucket_name = os.environ.get('AWS_S3_BUCKET', 'ferroallembeddings')
file_key = os.environ.get('EMBEDDINGS_KEY', 'embeddings.json')
file_name = os.environ.get('TRAINING_DATA_KEY', 'trainingData.pkl')



##Embedddings

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> List[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> Dict[tuple[str, str], List[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

def load_embeddings(fname: str) -> dict[tuple[str, str], List[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

# def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> List[(float, (str, str))]:
def order_document_sections_by_query_similarity(query: str, contexts: Dict[tuple[str, str], np.ndarray]) -> List[tuple[float, tuple[str, str]]]:

    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
        Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities



def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """

    MAX_SECTION_LEN = 500
    SEPARATOR = "\n* "
    ENCODING = "gpt2"  # encoding for text-davinci-003

    encoding = tiktoken.get_encoding(ENCODING)
    separator_len = len(encoding.encode(SEPARATOR))

    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    contx = "".join(chosen_sections)
    return contx, prompt
    
def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    conx, prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": 0.0,
        "max_tokens": 300,
        "model": COMPLETIONS_MODEL,
    }

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")





def getAnswer(myQuestion):
    # df = pd.read_pickle("trainingData.pkl")

    # with open('embeddings.json') as f:
    #     document_embeddings = json.load(f)
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    file_content = response['Body'].read().decode('utf-8')
    document_embeddings = json.loads(file_content)

    buffer = io.BytesIO()
    s3.download_fileobj(bucket_name, file_name, buffer)
    buffer.seek(0)
    df = pd.read_pickle(buffer)




    document_embeddings = {int(key): value for key, value in document_embeddings.items()}


    order_document_sections_by_query_similarity(myQuestion, document_embeddings)[:5]

    MAX_SECTION_LEN = 500
    SEPARATOR = "\n* "
    ENCODING = "gpt2"  # encoding for text-davinci-003

    encoding = tiktoken.get_encoding(ENCODING)
    separator_len = len(encoding.encode(SEPARATOR))

    f"Context separator contains {separator_len} tokens"




    conx, prompt = construct_prompt(
        myQuestion,
        document_embeddings,
        df
    )

    print(  "===\n", prompt)

    

    query = myQuestion
    answer = answer_query_with_context(query, df, document_embeddings)

    if (answer=="I don't know."):
        bullet_list = conx.split('*')
        bullet_list = [bullet for bullet in bullet_list if bullet and bullet.strip() != '']
        for i, phrase in enumerate(bullet_list):
            while phrase[0].isspace() or phrase[0]=='.' or phrase[0]==',' or len(phrase.split()[0]) == 1 or (len(phrase.split()[0]) == 2 and phrase.split()[0][1] == "."):
                phrase = phrase[1:].lstrip()
            bullet_list[i] = phrase[0].upper() + phrase[1:]

        # for i, phrase in enumerate(bullet_list):
        #     # Find the index of the first word
        #     first_word_index = phrase.find(' ')
        #     if first_word_index == -1:  # No spaces, so assume phrase is a single word
        #         first_word_index = len(phrase)
        #     # Remove any characters before the first letter and then remove leading/trailing whitespace
        #     for j in range(first_word_index):
        #         if phrase[j].isalpha():
        #             phrase = phrase[j:].strip()
        #             break
        #     # Remove single letters or single letters followed by a period at the beginning of the string
        #     while len(phrase) > 1 and (phrase[0].isspace() or phrase[0] == '.' or (len(phrase.split()[0]) == 1 and phrase.split()[0][-1] == '.')):
        #         phrase = phrase[1:].lstrip()
            # Capitalize the first letter
            # bullet_list[i] = phrase[0].upper() + phrase[1:]


        output_string = "\n\n".join(f"• {bullet.strip()}" for bullet in bullet_list)
        ans = "I don't know, but here is some context that may help.\n\n"+output_string
        return ans
    else:
        return answer
    

