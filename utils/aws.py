# Read AWS documentation
import os
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
from tqdm import tqdm
from markdown import markdown
from bs4 import BeautifulSoup
from haystack.schema import Document, Label, Answer, MultiLabel

# Custom Haystack conversion functions
from .haystack_pre_processing import markdown_to_documents

logger = logging.getLogger(__name__)

def aws_docs_import_qa(base_dir:str='./data/aws-documentation/') -> pd.DataFrame:
    "Import the local AWS documentation in to `pandas.DataFrame` from disk."
    df = pd.read_csv(base_dir+'/QA_true.csv')
    df.loc[df['Yes_No_True'].isnull(), 'Yes_No_True'] = None
    df['Answer_Full'] = [f'{y_n}, {answer}' if y_n is not None else answer for y_n, answer in zip(df['Yes_No_True'].tolist(), df['Answer_True'].tolist())]
    df['Document_True'] = df['Document_True'].str.strip()
    return df

def aws_docs_files(base_dir:str='./data/aws-documentation/documents') -> tuple[list[str], list[str]]:
    "Convert the local AWS documentation hierarchy into a list of file paths and names."
    folders = os.listdir(base_dir)
    logger.info(f"Reading {len(folders)} AWS documentation folders.")
    file_paths = []
    file_names = []
    for folder in tqdm(folders):
        src = os.path.join(base_dir, folder, 'doc_source')
        if os.path.isdir(src):
            md_files = [file for file in os.listdir(src) if file.endswith('.md')]
            for file in md_files:
                f = os.path.join(base_dir, folder, 'doc_source', file)
                file_paths.append(f)
                doc = folder+'/'+file
                file_names.append(doc.strip())
    return file_paths, file_names

def aws_docs_documents(base_dir:str='./data/aws-documentation/', modalities:list[str]=['text','table']) -> list[Document]:
    "Convert the local AWS documentation hierarchy of markdown files into a list of `Document` objects from the `haystack` library."
    if len(modalities) > 1:
        logger.warning("You are extracting multiple modalities, make sure you have a multi-modal approach.")
    paths, names = aws_docs_files(base_dir+'documents')
    documents = []
    n = len(paths)
    if 'table' not in modalities and 'text' not in modalities:
        raise ValueError('At least one of `table` or `text` must be in `modalities`.')
    elif 'table' not in modalities:
        logger.info('Parsing tables to `text`')
    else:
        logger.info('Parsing tables to `pandas.DataFrame()`')
        logger.warning('Multi-Modality must be supported by the reader.')
    logger.info(f"Converting {n} AWS documentation files to `Document` objects with modalities: `{' - '.join(modalities)}`.")
    for p, n in tqdm(zip(paths, names), total=n):
        with open(p, 'r', encoding='utf-8') as f:
            md = f.read()
        documents += markdown_to_documents(md, meta={'name':n}, modalities=modalities)
    return documents

def aws_docs_labels(documents:list[Document], base_dir:str='./data/aws-documentation/') -> list[Label]:
    "Read the local AWS documentation answers and convert them in to haystack `Label` objects (we need the haystack `Document` that correspond to the label)."
    df_true = aws_docs_import_qa(base_dir=base_dir)
    #questions = df_true['Question'].tolist()
    documents_true = df_true['Document_True'].tolist()
    answer_cols = ['Answer_True', 'Yes_No_True', 'Answer_Full']
    label_documents = [[doc for doc in documents if doc.meta.get('name') in d][0] for d in tqdm(documents_true)]
    labels = []
    # We iterate over the dataframe rows containing the questions and answers + the haystack documents
    for (idx, row), document_true_haystack in zip(df_true.iterrows(), label_documents):
        _ = []
        # We create a label for each answer column (we have 3 answer columns, so 3 labels per question) 
        for col in answer_cols:
            question, answer_true, document_true = row['Question'], row[col], row['Document_True']
            if answer_true is None:
                logger.warning(f"Question: {question} has no answer in column: {col}, skipping.")
            else:
                label = Label(query=question, answer=Answer(answer_true, 'other'), document=document_true_haystack, 
                        is_correct_answer=True, is_correct_document=True, origin='gold-label')
                label.created_at=time.strftime("%Y-%m-%d %H:%M:%S")  
                _.append(label)
                labels.append(label)
    logger.info(f"Created {len(labels)} labels from {len(df_true)} questions.")
    return labels
    
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    md = open('./data/aws-documentation/documents/amazon-ecr-user-guide/doc_source/doc-history.md', 'r', encoding='utf-8').read()
    docs = markdown_to_documents(md, meta={'name':'amazon-ecr-user-guide/doc-history.md'}, modalities=['text'])
    #docs = soup_to_documents('./data/aws-documentation/documents/amazon-ecr-user-guide/doc_source/doc-history.md', meta={'name':'amazon-ecr-user-guide/doc-history.md'})
    len(docs)
    docs[0]
    docs[1]
    
    # Local files
    paths, names = aws_docs_files() # the local AWS documentation hierarchy
    documents = aws_docs_documents(modalities=['text']) # documents converted to haystack Documents
    labels = aws_docs_labels(documents) # the answers to the questions converted to haystack Labels
    df_true = aws_docs_import_qa() # question and answers in a pandas DataFrame
    assert len(labels)==len(df_true), 'The number of imported haystack `Document` labels must be equal to the question/answer dataframe length.'
    
    # Questions and answers
    df_true['Question'].tolist()
    df_true['Document_True'].tolist()
    i = 1
    df_true.iloc[i, :]['Question']
    df_true.iloc[i, :]['Answer_True']
    df_true.iloc[i, :]['Document_True']
    
    
    
    
    
    
        # Haystack sample
    # Document store based on AWS documentation
    from haystack.document_stores import InMemoryDocumentStore
    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.write_documents(documents)
    document_store.write_labels(labels)
    
    # Retriever
    from haystack.nodes import BM25Retriever
    retriever = BM25Retriever(document_store=document_store)
    # Reader
    from haystack.nodes import FARMReader
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    # Pipeline
    from haystack.pipelines import ExtractiveQAPipeline
    pipe = ExtractiveQAPipeline(reader, retriever)

    # Query
    query = 'Is Amazon EBS encryption available on M3 instances?' #df_true.iloc[0, :]['Question']
    answer = 'Amazon EBS encryption is available on all current generation instance types and the following previous generation instance types: C3 cr1.8xlarge G2 I2 M3 and R3' #df_true.iloc[0, :]['Answer_True']
    document_true = 'amazon-ec2-user-guide/EBSEncryption.md' #df_true.iloc[0, :]['Document_True']
    document_true
    print(f'Question: {query} \nAnswer: {answer}')
    result = pipe.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
    result['documents'].keys()
    from haystack.utils import print_answers
    print_answers(result, details="minimal")
    
    
    
    QUESTIONS = df_true['Question'].tolist()
    ANSWERS = df_true['Answer_True'].tolist()
    DOCUMENTS = df_true['Document_True'].tolist()
    