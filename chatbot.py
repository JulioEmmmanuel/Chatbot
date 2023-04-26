import nltk
import ssl
import pandas as pd
import re

from nltk.stem import wordnet                                # to perform lemmitization
from nltk import pos_tag                                       # for parts of speech
from sklearn.metrics import pairwise_distances                 # to perfrom cosine similarity
from sklearn.feature_extraction.text import CountVectorizer    # to perform bow
from nltk import word_tokenize                                 # to create tokens
from nltk.corpus import stopwords   # for stop words

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')                    # uncomment if running the cell for the first time
nltk.download('w')
nltk.download('averaged_perceptron_tagger')      # uncomment if running the cell for the first time
nltk.download('stopwords')            # uncomment if running the cell for the first time

df = pd.read_csv("mentalhealth.csv", nrows = 20)
df.head()

# function that performs text normalization steps and returns the lemmatized tokens as a sentence
stop = stopwords.words('english')

def text_normalization(text):
    text = str(text).lower()                        # text to lower case
    spl_char_text = re.sub(r'[^ a-z]','',text)      # removing special characters
    tokens = word_tokenize(spl_char_text)      # word tokenizing
    lema = wordnet.WordNetLemmatizer()              # intializing lemmatization
    tags_list = pos_tag(tokens,tagset=None)         # parts of speech
    lema_words = []                                 # empty list 
    for token,pos_token in tags_list:               # lemmatize according to POS
        if pos_token.startswith('V'):               # Verb
            pos_val = 'v'
        elif pos_token.startswith('J'):             # Adjective
            pos_val = 'a'
        elif pos_token.startswith('R'):             # Adverb
            pos_val = 'r'
        else:
            pos_val = 'n'                           # Noun
        lema_token = lema.lemmatize(token,pos_val)

        if lema_token in stop: 
          lema_words.append(lema_token)             # appending the lemmatized token into a list
    
    return " ".join(lema_words) 

df['lemmatized_text'] = df['Questions'].apply(text_normalization)   # clean text
cv = CountVectorizer()                                  # intializing the count vectorizer
X = cv.fit_transform(df['lemmatized_text']).toarray()

# returns all the unique word from data 
features = cv.get_feature_names_out()
df_bow = pd.DataFrame(X, columns = features)

# defining a function that returns response to query using bow
def chat_bow(text):
    lemma = text_normalization(text) # calling the function to perform text normalization
    bow = cv.transform([lemma]).toarray() # applying bow
    cosine_value = 1- pairwise_distances(df_bow,bow, metric = 'cosine' )
    index_value = cosine_value.argmax() # getting index value 
    return df['Answers'].loc[index_value]
