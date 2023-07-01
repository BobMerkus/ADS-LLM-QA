# DEPENDENCIES
import os
import requests
import time
import random
import logging
import json
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

# CONSTANTS
IMAGE_FORMATS = ("png", "jpeg", "jpg", "gif", "svg", "bmp", "tiff", "tif")
VIDEO_FORMATS = ("mp4", "avi", "mkv", "mov", "flv", "wmv", "webm")
AUDIO_FORMATS = ("mp3", "wav", "ogg", "flac", "aac", "wma", "m4a")

def url_to_local_directory(url:str, local_directory:str) -> str:
    "Returns the local directory for a given URL"
    parsed_url = urlparse(url)
    # Create the local directory if it doesn't exist
    if not os.path.exists(local_directory):
        os.mkdir(local_directory)
    base_dir = os.path.join(local_directory, parsed_url.netloc)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # create subdirectories based on url
    segments = parsed_url.path.split('/')
    to_create = [dir for dir in segments if dir]
    if len(to_create) > 0:
        current = base_dir
        for i, dir in enumerate(to_create):
            if not i+1 == len(to_create):
                current+=f'/{dir}'
                if not os.path.exists(current):
                    os.mkdir(current)
            else:
                filepath = os.path.join(current, dir + '.html')
                return filepath
    else:
        return base_dir + '/index.html'

# FUNCTIONS
def get_proxies_from_file(dir:str) -> dict:
    "Returns a dictionary of proxies from the files in the specified directory"
    files = ['http.txt', 'socks4.txt', 'socks5.txt']
    proxies = {file.split('.')[0]: [line.strip() for line in open(dir + file, 'r').readlines()] for file in files}
    return proxies

def parse_pdf(data) -> str:
    "Returns the text from a pdf file"
    print('Parsing pdf file...')
    import PyPDF2
    import io
    file = io.BytesIO(data)
    # Open the PDF file in read binary mode
    pdf_reader = PyPDF2.PdfReader(file)
    # Get the total number of pages in the PDF file
    num_pages = pdf_reader.pages
    # Iterate through each page and extract the text
    text = ''
    for i in range(len(num_pages)):
        page = pdf_reader.pages[i]
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get(url:str, local_directory:str, proxies:dict=None, max_sleep:int=3, sleep_multiplier:int=5, cache:bool=True) -> BeautifulSoup:
    """_summary_
    Downloads a web page from a URL, or loads it from the local cache if available.  Returns the web page content as a BeautifulSoup object.

    Args:
        url (str): 
        local_directory (str): 
        proxies (dict, optional): Defaults to None.
        max_sleep (int, optional): Defaults to 3.
        sleep_multiplier (int, optional): Defaults to 5.
        verbose (bool, optional): Defaults to True.

    Returns:
        BeautifulSoup: 
    """
    
    logger.info(f'Getting `{url}`...')
    # Use the URL as the filename (with the appropriate extension)
    filepath = url_to_local_directory(url, local_directory=local_directory)

    # Return image links + video links as is
    if any([url.endswith(f) for f in IMAGE_FORMATS + VIDEO_FORMATS]):
        data = url
    # Pdf files are parsed to text
    elif url.endswith('.pdf') :
        response = requests.get(url, proxies=proxies)
        data = parse_pdf(response.content)
    # Parse html files
    else:
        # Check if the file is already in the cache
        if os.path.exists(filepath) and cache:
            logger.info(f'Loading `{url}` from cache')
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        # Download the file and save it to the cache
        else:
            if proxies is None:
                logger.warning(f'No proxies were supplied, attempting to use random proxies from `./data/proxies/`...')
                proxies = get_proxies_from_file('./data/proxies/')
            proxies = {k:random.choice(v) for k, v in proxies.items()}
            logger.info(f'Downloading using proxies {proxies}')
            response = requests.get(url, proxies=proxies)
            # Respect the server's wishes by waiting before downloading again
            response_time = response.elapsed.total_seconds()
            sleep_time = response_time*sleep_multiplier if response_time*sleep_multiplier < max_sleep else max_sleep
            logger.info(f"Response time was: {response_time}. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            content = response.content.decode('utf-8')
            os.makedirs(local_directory, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        # Return the content as a BeautifulSoup object
        data = BeautifulSoup(content, 'html.parser')
    return data


def webcrawl(domain:str, local_directory:str, target_subdomain:str='', 
             max_pages:int=50, checkpoint:int=50, checkpoint_file:str='./data/checkpoint.json',
             warmup:int=10, max_sleep:int=3, sleep_multiplier:int=5,
             proxies:dict=None) -> tuple:
    """_summary_

    Args:
        - domain (str):  Crawl the web for a given domain. 
        - local_directory (str): . Save the files to this local directory
        - target_subdomain (str, optional): . Recursively crawl all links found in the target_subdomain. Defaults to ''.
        - max_pages (int, optional): . Maximum number of pages to download. Defaults to 50.
        - checkpoint (int, optional): . Save the parsed text .json every `n` webpages. Defaults to 50.
        - warmup (int, optional): . Force download the first `n` pages to discover targeted subdomains. Defaults to 10.
        - max_sleep (int, optional): . Maximum sleep time in seconds before making another request. Defaults to 3.
        - proxies (dict, optional): . Proxies to use for this session. Highly recommended and will therefore raise warning without. Defaults to None.

    Returns:
        _type_: 
    """
    visited = set()
    data = []
    queue = [domain]
    queue_history = []
    i = 0
    while queue and len(visited) < max_pages:
        target = domain + target_subdomain
        url = queue.pop(0)
        # only crawl the target subdomain or if no data has been collected
        if url not in visited and (target in url or len(data)<warmup): 
            logger.info(f'Queue: {len(queue)} | Visited: {len(visited)}')
            soup = get(url, proxies=proxies, local_directory=local_directory, max_sleep=max_sleep, sleep_multiplier=sleep_multiplier)
            # Extract all links from the page
            links = [link.get('href') for link in soup.find_all('a')]
            for link in links:
                if link is None:
                    continue
                if link.startswith('/'):
                    link = domain + link[1:]
                if not link.startswith('http'):
                    continue
                link = link.split('#')[0] # remove anchors
                queue.append(link)
            logger.info(f'Found {len(links)} links')
            # Update info and sleep
            visited.add(url)
            if soup.title is not None:
                title = soup.title.string
            else:
                title = ''
            # result
            selected_elements = soup.find_all('p') #['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            text = ' '.join([e.get_text().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip() for e in selected_elements])         
            record = {'url': url, 'text': text, 'title': title, 'text_length': len(text), 'title_length': len(title)}
            data.append(record)
            queue_history.append(len(queue))
            print(f"Succesfully crawled {title}: {url}")
            # Checkpoint
            i += 1
            if i % checkpoint == 0:
                logger.info('Checkpoint: Saving data...')
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            logger.info(f'Skipping {url}')
    # Save data
    df_queue = pd.DataFrame(queue_history, columns=['queue_count'])
    df_queue['iteration'] = df_queue.index
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return visited, data, df_queue
    


if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    url = 'https://kubernetes.io/docs/concepts/overview'
    url_to_local_directory(url, './data/bot_memory') #clone to local dir
     
    soup = get(url, local_directory='./data/bot_memory/')
    
        
    visited, data, df_queue = webcrawl(
        domain = 'https://kubernetes.io/', 
        local_directory = 'data/bot_memory/',
        target_subdomain='docs',
        max_pages=25,
        max_sleep=2
    )   
    
    
    