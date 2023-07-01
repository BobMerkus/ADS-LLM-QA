
import time
import logging
import os

logger = logging.getLogger(__name__)

def get_key(file:str):
    "Function to import .txt key files with try catch and logging."
    try:
        with open(file, 'r') as f:
            key = f.readline()
            if not key:
                logger.warning(f"No valid key found in {file}")
    except FileNotFoundError:
        logger.error(f"FileNotFoundError: {file} does not exist")
        key = ''
    return key

def add_time_elapsed(func):
    "Function to add time elapsed to a function that returns a dictionary. Should be used as a decorator."
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        result['time_elapsed'] = elapsed_time
        return result
    return wrapper

def get_color(red:int=255,green:int=255,blue:int=255) -> str:
    "Function to turn r,g,b in to hex string"
    color = (red, green, blue)
    if max(color)<=1:
        color = (c * 255 for c in color)
    color = (int(c) for c in color)
    color = '#{:02x}{:02x}{:02x}'.format(*color)
    return color

def elasticsearch(port:int=9200, es_version:str="8.8.1"):
    "Use Docker to start an Elasticsearch instance with a single node."
    logger.info(f"Starting Elasticsearch ({es_version}) on port {port}...")
    os.system(f"docker pull docker.elastic.co/elasticsearch/elasticsearch:{es_version}")
    os.system(f"docker network create elastic")
    os.system(f"docker run --name es01 --net elastic -p {port}:{port} -it docker.elastic.co/elasticsearch/elasticsearch:{es_version}")
    if not os.path.exists('./data/ca.crt'):
        logger.error('No certificate found, attempting to copy from container...')
        os.system('docker cp es01:/usr/share/elasticsearch/config/certs/ca/ca.crt ./data')
