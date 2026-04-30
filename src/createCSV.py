import re
from typing import Set
from transformers import GPT2TokenizerFast
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
import openai
import pickle
import tiktoken
from typing import List, Dict
from PyPDF2 import PdfReader
import PyPDF2
import os
import spacy
import io
import docx2txt 
import subprocess
import pdf2docx
import shutil
import json
import time


os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
openai.api_key = os.environ.get("OPENAI_API_KEY")


def is_pdf(file_path):
    with open(file_path, 'rb') as file:
        header = file.read(4)
        if header == b'%PDF':
            return True
        else:
            return False

def is_doc_or_docx(file_path):
    with open(file_path, 'rb') as file:
        header = file.read(8)

    if header[:4] == b'PK\x03\x04':
        return True
    elif header.startswith(b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1') or header.startswith(b'\x00\x6E\x1E\xF0'):
        return True
    else:
        return False



##Preprocessing the Data - converts all files in 'trainingDataPdfs' to docxs and stores them in 'trainingDataDocs'
def preprocess():
    processed = []
    # Set the directory path
    dir_path = 'trainingDataDocs'
    # Get a list of all files in the directory
    for root, dirs, files in os.walk(dir_path):
        # Iterate over all files within the current directory
        for file in files:
            fname = os.path.basename(os.path.join(root, file))
            if fname != '.DS_Store':
            # Record all file names in this directory
                processed.append(fname)


    # Set the directory path
    dir_path = 'trainingDataPdfs'
    # Get a list of all files in the directory
    for root, dirs, files in os.walk(dir_path):
        # Iterate over all files within the current directory
        for file in files:
            fname = dir_path+'/'+file
            #Convert pdfs to docx and store in 'trainingDataDocs'
            if is_pdf(fname):
                if fname[-4::]=='.pdf':
                    altered = fname[0:-4]
                if altered not in processed:
                    pdfToText(file, root)
            elif is_doc_or_docx(fname):
                # original_file_path = root+'/'+fname
                new_file_path = 'trainingDataDocs/'+file
                shutil.copyfile(fname, new_file_path)

# Converts pdfs to docxs and stores in 'trainingDataDocs'
def pdfToText(fileName, root):

    pdf_path = root+'/'+fileName

    if fileName[-4::]=='.pdf':
        fileName = fileName[0:-4]+'.docx'
    docx_path = 'trainingDataDocs/'+fileName  
    
    pdf2docx.parse(pdf_path, docx_path)
    

# From all files in 'trainingDataDocs', create training data
def extractData(fileName):

    # Set Title of data to be aquired
    title = fileName

    docx_path = 'trainingDataDocs/'+fileName

    # Extract text from docxs files
    fullText = docx2txt.process(docx_path)
    
    # Determine section headers
    keywords = ['abstract', 'introduction', 'experiment', 'results', 'discussion', 'conclusion', 'concluding remarks', 'references', 'acknowledgment', 'conflict of interest', 'keywords', 'acknowledgement', 'content']

    # lowerFullText = fullText.lower()

    # Separate text into (keyword, content) pairs
    # pattern = '|'.join(map(re.escape, keywords))
    # sections = [(kw, section.strip()) for kw, section in re.findall(f'({pattern})(.*?)(?={pattern}|$)', lowerFullText, re.DOTALL)]

    # Remove content sections with useless keywords
    # pattern = '|'.join(map(re.escape, keywords))
    # sections = [(kw, section.strip()) for kw, section in re.findall(f'({pattern})(.*?)(?={pattern}|$)', fullText, flags=re.IGNORECASE|re.DOTALL)]

    # sections = [sec for sec in sections if (sec[0] != 'references' and sec[0] != 'acknowledgment' and sec[0] != 'acknowledgement' and sec[0] != 'conflict of interest' and sec[0] != 'keywords')]

    pattern = '|'.join(map(re.escape, keywords))
    unwanted_keywords = ["references", "acknowledgment", "acknowledgement", 'conflict of interest', 'keywords', 'content']
    sections = [(kw, section.strip()) for kw, section in re.findall(f'({pattern})(.*?)(?={pattern}|$)', fullText, flags=re.IGNORECASE|re.DOTALL) if kw.lower() not in [w.lower() for w in unwanted_keywords]]
    
# Clean up data
    newSections = []
    for sec in sections:
        tokenCount = count_tokens(sec[1])
        if tokenCount > 1500:
            reduced = reduce_long(sec[1], 1500)
            tokenCount = count_tokens(reduced)
        else:
            reduced = sec[1]

        reduced = reduced.replace("\n", "")
        reduced = reduced.replace("\t", "")
        reduced = re.sub(' +', ' ', reduced)

        # (keyword, content) -> (title, keyword, content, tokenCount)
        newTup = (title, sec[0], reduced, tokenCount)
        newSections.append(newTup)

    # Return [(title, keyword, content, tokenCount), ... (title, keyword, content, tokenCount)]
    return newSections


# Count the number of tokens in a string
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
# def reduce_long(
#     long_text: str, long_text_tokens: bool = False, max_len: int = 590
# ) -> str:

#     if not long_text_tokens:
#         long_text_tokens = count_tokens(long_text)
#     if long_text_tokens > max_len:
#         sentences = sent_tokenize(long_text.replace("\n", " "))
#         ntokens = 0
#         for i, sentence in enumerate(sentences):
#             ntokens += 1 + count_tokens(sentence)
#             if ntokens > max_len:
#                 return ". ".join(sentences[:i][:-1]) + "."

#     return long_text
# Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
def reduce_long(long_text: str, max_len: int = 1500) -> str:
    long_text_tokens = count_tokens(long_text)
    if long_text_tokens <= max_len:
        return long_text
    
    # Split the long text into sentences
    sentences = sent_tokenize(long_text.replace("\n", " "))
    
    # Join the sentences until the token count is just under the maximum
    reduced_sentences = []
    reduced_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if reduced_tokens + sentence_tokens > max_len:
            break
        reduced_sentences.append(sentence)
        reduced_tokens += sentence_tokens + 1  # Add one for the period after the sentence
    
    return ". ".join(reduced_sentences) + "."




def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> List[float]:
    try:
        result = openai.Embedding.create(
          model=model,
          input=text
        )
    except openai.error.RateLimitError as e:
        # print(f"Rate limit reached. Waiting for 1 seconds.")
        time.sleep(1)
        try:
            result = openai.Embedding.create(
              model=model,
              input=text
            )
        except openai.error.RateLimitError as e:
            print(f"Rate limit reached. Waiting for 10 seconds.")
            time.sleep(10)
            try:
                result = openai.Embedding.create(
                  model=model,
                  input=text
                )
            except openai.error.RateLimitError as e:
                print(f"Rate limit reached. Waiting for 20 seconds.")
                time.sleep(20)
                try:
                    result = openai.Embedding.create(
                      model=model,
                      input=text
                    )
                except openai.error.RateLimitError as e:
                    print(f"Rate limit reached. Waiting for 30 seconds.")
                    time.sleep(30)
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



#Run the code


preprocess()

def create():
    allData = []
    dir_path = 'trainingDataDocs'
    # Get a list of all files in the directory
    for root, dirs, files in os.walk(dir_path):
        # Iterate over all files within their current directory
        for file in files:
            if file != '.DS_Store':
                allData+=extractData(file)


    df = pd.DataFrame(allData, columns=["title", "heading", "content", "tokens"])
    df = df[df.tokens>40]
    # df = df.drop_duplicates(['title','heading'])
    df = df.reset_index().drop('index',axis=1) # reset index
    df.head()

    # try:
    #     currdf = pd.read_pickle("trainingData.pkl")
    #     frames = [currdf, df]
    #     df = pd.concat(frames)

    #     # df = currdf.add(df)
    # except:
    #     pass
        
    df.to_csv('trainingData.csv', index=False)

    df.to_pickle("trainingData.pkl")
    return df

df = create()
document_embeddings = compute_doc_embeddings(df)
with open('embeddings.json', 'w') as f:
    json.dump(document_embeddings, f)

