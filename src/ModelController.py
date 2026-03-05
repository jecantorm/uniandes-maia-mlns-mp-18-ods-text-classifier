# src/ModelController.py
import os
import re
import nltk
import joblib
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

JOBLIB_PIPELINE_PATH = "pipeline_quick_balanced_min_df2.joblib"

for _pkg, _path in [('stopwords', 'corpora/stopwords'), ('punkt', 'tokenizers/punkt')]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

SPANISH_STOPWORDS = set(stopwords.words('spanish'))
_tokenizer = RegexpTokenizer(r'\w+')
_stemmer = SnowballStemmer("spanish")


def text_preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)
    tokens = _tokenizer.tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2 and t not in SPANISH_STOPWORDS]
    stems = [_stemmer.stem(t) for t in tokens]
    return " ".join(stems)

class ModelController:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..", "resources", "models", JOBLIB_PIPELINE_PATH
            )
        self.pipeline = joblib.load(model_path)

    def predict(self, texts: list) -> list:
        return self.pipeline.predict(texts).tolist()