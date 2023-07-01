import logging
from bs4 import BeautifulSoup
from markdown import markdown
import pandas as pd
from haystack.schema import Document

logger = logging.getLogger(__name__)

def soup_to_documents(soup:BeautifulSoup, meta:dict=None, modalities:list[str]=['text','table']) -> list[Document]:
    "Convert a bs4 soup into a list of `Document` objects from the `haystack` library. Extracts <table> elements and optionally converts the extracted tables to text."
    docs = []
    tables = soup.findAll('table')
    for table in tables:
        t = soup.extract(table)
        try:
            dfs = pd.read_html(str(t))
            for df in dfs:
                if df.shape[0] > 0:
                    if 'table' in modalities:
                        doc = Document(content=df, meta=meta, content_type='table')
                    if 'text' in modalities:
                        records = df.to_dict(orient='records')
                        text = [', '.join([f'{k}: {v}' for k, v in rec.items()]) for rec in records]
                        text = '.\n'.join(text)
                        doc = Document(content=text, meta=meta, content_type='text')
                        docs.append(doc)
            soup.table.decompose()
        except Exception as e:
            logger.error(f'Error converting table to text: {e}')            
    if 'text' in modalities:
        text = ''.join(soup.findAll(string=True)) 
        if len(text) > 0:
            doc = Document(content=text, meta=meta, content_type='text')
            docs.append(doc)                        
    return docs

def markdown_to_documents(md:str, meta:dict=None, modalities:list[str]=['text','table']) -> list[Document]:
    "Convert a markdown string into a list of `Document` objects from the `haystack` library. Extracts <table> elements and optionally converts the extracted tables to text."
    html = markdown(md, extensions=['markdown.extensions.tables'])
    soup = BeautifulSoup(html, 'html.parser')
    return soup_to_documents(soup, meta=meta, modalities=modalities)