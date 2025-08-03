"""
data.py: Dataset loading, preprocessing, and prompt construction for FACTER.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import zipfile
import gzip
import shutil
import json
from tqdm import tqdm
from .config import Config
import logging

logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Loads and preprocesses MovieLens and Amazon data. 
    Aligned with sections on Datasets (Section 4.2).
    """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = None
        self.item_db = None
        self._load_dataset()
        
    def _load_dataset(self):
        if self.dataset_name == 'ml-1m':
            self._load_movielens()
        elif self.dataset_name == 'amazon':
            self._load_amazon()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _load_movielens(self):
        try:
            self._download_movielens()
            ratings = pd.read_csv(
                Config.EXTRACT_DIR/'ml-1m'/'ratings.dat',
                sep='::', engine='python', 
                names=['uid','mid','rating','timestamp']
            )
            users = pd.read_csv(
                Config.EXTRACT_DIR/'ml-1m'/'users.dat',
                sep='::', engine='python', 
                names=['uid','gender','age','occupation','zip']
            )
            movies = pd.read_csv(
                Config.EXTRACT_DIR/'ml-1m'/'movies.dat',
                sep='::', engine='python', 
                names=['mid','title','genre'],
                encoding='latin-1'
            )
            self.data = ratings.merge(users, on='uid').sort_values(['uid','timestamp'])
            self.item_db = movies.set_index('mid').to_dict(orient='index')
        except Exception as e:
            logger.error(f"MovieLens loading failed: {str(e)}")
            raise

    def _download_movielens(self):
        if not (Config.EXTRACT_DIR/'ml-1m').exists():
            try:
                response = requests.get(Config.DATASETS['ml-1m']['url'], timeout=30)
                response.raise_for_status()
                with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(Config.EXTRACT_DIR)
            except Exception as e:
                logger.error(f"MovieLens download failed: {str(e)}")
                raise

    def _load_amazon(self):
        try:
            self._download_amazon()
            file_path = Config.EXTRACT_DIR / 'Movies_and_TV_5.json.gz'
            records = []
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading Amazon data"):
                    records.append(json.loads(line))
            self.data = pd.DataFrame(records)
            self._preprocess_amazon()
        except Exception as e:
            logger.error(f"Amazon dataset loading failed: {str(e)}")
            raise

    def _download_amazon(self):
        file_path = Config.EXTRACT_DIR / 'Movies_and_TV_5.json.gz'
        if not file_path.exists():
            try:
                logger.info("Downloading Amazon dataset...")
                response = requests.get(
                    Config.DATASETS['amazon']['url'], 
                    stream=True, 
                    verify=False   
                )
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192), 
                                    desc="Downloading", unit="KB",
                                    total=int(response.headers.get('content-length', 0))/8192):
                        f.write(chunk)
                logger.info("Extracting dataset...")
                with gzip.open(file_path, 'rb') as f_in:
                    with open(Config.EXTRACT_DIR / 'Movies_and_TV_5.json', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                logger.error(f"Amazon download failed: {str(e)}")
                raise

    def _preprocess_amazon(self):
        self.data = self.data[self.data['overall'] >= 4]
        self.data = self.data.rename(columns={
            'reviewerID': 'uid',
            'asin': 'mid',
            'reviewText': 'text',
            'overall': 'rating'
        })
        np.random.seed(42)
        self.data['gender'] = np.random.choice(['M','F'], size=len(self.data))
        self.data['age'] = np.random.randint(18, 65, size=len(self.data))
        self.data['occupation'] = np.random.choice(20, size=len(self.data))
        self.data['timestamp'] = self.data['unixReviewTime']
        self.data.rename(columns={'summary': 'title'}, inplace=True)
        self.item_db = (
            self.data
            .drop_duplicates('mid')
            .set_index('mid')[['title']]
            .to_dict(orient='index')
        )

    def prepare_prompts(self):
        try:
            self.data = self.data.sort_values(['uid', 'timestamp'])
            self.data['sequence'] = self.data.groupby('uid')['mid'].transform(
                lambda x: [x.iloc[:i].tolist()[-5:] for i in range(len(x))]
            )
            self.data = self.data[self.data['sequence'].apply(
                lambda x: isinstance(x, list) and len(x) >= 3
            )]
            if self.dataset_name == 'ml-1m':
                self.data['prompt'] = self.data['sequence'].apply(
                    lambda seq: self._create_movielens_prompt(seq)
                )
            else:
                self.data['prompt'] = self.data['sequence'].apply(
                    lambda seq: self._create_amazon_prompt(seq)
                )
            return self.data[['prompt','gender','age','occupation','mid']].dropna()
        except Exception as e:
            logger.error(f"Prompt generation failed: {str(e)}")
            raise

    def _create_movielens_prompt(self, sequence):
        try:
            history = [self.item_db[mid]['title'] for mid in sequence[:-1]]
            candidates = [self.item_db[mid]['title'] for mid in sequence[-3:]]
            return (
                "Movie watching history:\n" +
                '\n'.join([f"{i+1}. {m}" for i, m in enumerate(history)]) +
                "\n\nRecommend next movie from these options:\n" +
                '\n'.join([f"{i+1}. {m}" for i, m in enumerate(candidates)])
            )
        except KeyError as e:
            logger.warning(f"Missing movie ID in database: {str(e)}")
            return None

    def _create_amazon_prompt(self, sequence):
        try:
            history = [self.item_db[mid]['title'] for mid in sequence[:-1]]
            candidates = [self.item_db[mid]['title'] for mid in sequence[-3:]]
            return (
                "Product interaction history:\n" +
                '\n'.join([f"{i+1}. {m}" for i, m in enumerate(history)]) +
                "\n\nRecommend next product from these options:\n" +
                '\n'.join([f"{i+1}. {m}" for i, m in enumerate(candidates)])
            )
        except KeyError as e:
            logger.warning(f"Missing product ID in database: {str(e)}")
            return None
