import logging
# constants
from .webscraper import IMAGE_FORMATS, VIDEO_FORMATS, AUDIO_FORMATS
# functions
from .nlp import hash_md5, match, split_chunks, normalize_answer, f1_score
#from .nlp import rouge, rouge_n, rouge_l, rouge_w
from .webscraper import get_proxies_from_file, url_to_local_directory, parse_pdf, get, webcrawl
from .stackexchange import read_stackexchange, get_stackexchange, stackexchange_documents
from .aws import aws_docs_import_qa, aws_docs_files, aws_docs_documents, aws_docs_labels
from .kubernetes import kubernetes_documents
from .misc import get_key, add_time_elapsed, get_color
from .haystack_pre_processing import soup_to_documents, markdown_to_documents