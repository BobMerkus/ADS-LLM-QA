import json
from haystack.schema import Document

def kubernetes_documents(file_path:str='./data/kubernetes_docs.json'):
    "Convert a local (webscraped) json file containing kubernetes data to a haystack document"
    data = json.load(open(file_path, 'r'))
    documents = []
    for d in data:
        d['name'] = d.pop('title')
        content = d.pop('text')
        documents.append(Document(content=content, meta=d, content_type='text'))
    return documents