import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from typing import List, Tuple, Dict
from collections.abc import Callable
import numpy as np
from numpy import linalg as LA
import json
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds



#test
# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

"""
Loading in data -- Start
"""
# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
# with open(json_file_path, 'r') as file:
#     data = json.load(file)
#     product_df = pd.DataFrame(data['Product Name'])
#     description_df = pd.DataFrame(data['Product Description'])
#     link_df = pd.DataFrame(data['Product URL'])
f = open(json_file_path)
data = json.load(f)
print("JSON succesfully loaded!")
f.close()

"""
Loading in data -- Finish
"""

app = Flask(__name__)
CORS(app)

from typing import List, Tuple, Dict
from collections.abc import Callable
import numpy as np
import re
from typing import List, Tuple, Dict
from collections.abc import Callable
import numpy as np
from numpy import linalg as LA
import json
import math


"""
Cosine Similarity Calculation Functions -- Start
"""
def tokenize(text: str) -> List[str]:
    """Returns a list of words that make up the text.
    
    Note: for simplicity, lowercase everything.
    Requirement: Use Regex to satisfy this function
    
    Parameters
    ----------
    text : str
        The input string to be tokenized.

    Returns
    -------
    List[str]
        A list of strings representing the words in the text.
    """
    text = text.lower()
    return re.findall(r'[a-z]+', text)

def build_inverted_index(msgs:dict) -> dict:
    """Builds an inverted index from the messages.

    Arguments
    =========

    msgs: list of dicts.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    Example
    =======

    >> test_idx = build_inverted_index([
    ...    {'toks': ['to', 'be', 'or', 'not', 'to', 'be']},
    ...    {'toks': ['do', 'be', 'do', 'be', 'do']}])

    >> test_idx['be']
    [(0, 2), (1, 2)]

    >> test_idx['not']
    [(0, 1)]

    """
    solution = {}
    for i in msgs:
      tokens = msgs[i]
      for token in tokens:
        if token in solution:
          if i in solution[token]:
            solution[token][i]+=1
          else:
            solution[token][i] = 1
        else:
          solution[token] = {}
          solution[token][i] = 1
    result = {}
    for key in solution:
      temp = []
      for key2 in solution[key]:
        temp.append((key2,solution[key][key2]))
      temp.sort(key=lambda x:x[0])
      result[key] = temp
    return result

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    Hint: Make sure to use log base 2.

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """
    import math
    solution = {}
    for word in inv_idx:
      if len(inv_idx[word]) >= min_df and (len(inv_idx[word])/n_docs) <= max_df_ratio:
        calc = n_docs/(1 + len(inv_idx[word]))
        calc = math.log2(calc)
        
       
        solution[word] = calc
    

    return solution

def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document.
    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.
    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    import math
    solution = [0] * n_docs
    doc_sums = {}
    for word in index:
      for pair in index[word]:
        if word in idf:
          if pair[0] in doc_sums:
            doc_sums[pair[0]] += (pair[1] * idf[word]) ** 2
          else:
            doc_sums[pair[0]] = (pair[1] * idf[word]) ** 2
        
    for i in range(n_docs):
      if i in doc_sums:
        solution[i] = math.sqrt(doc_sums[i])
    return solution

def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========

    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    doc_sums = {}
    for word in index:
      if word in query_word_counts and query_word_counts[word]!=0 and word in idf:
        for pair in index[word]:
          if pair[1]!=0:
            if pair[0] in doc_sums:
              doc_sums[pair[0]] += (query_word_counts[word]*pair[1]*idf[word]*idf[word])
            else:
              doc_sums[pair[0]] = (query_word_counts[word]*pair[1]*idf[word]*idf[word])
        
    return doc_sums

def index_search(
    query: str,
    index: dict,
    idf,
    doc_norms,
    score_func=accumulate_dot_scores,
    tokenizer=tokenize,
) -> List[Tuple[int, int]]:
    """Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.

    index: an inverted index as above

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)

    tokenizer: a TreebankWordTokenizer

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.

    Note:

    """

    # TODO-8.1
    import math
    query_tokens = tokenizer(query.lower())
    query_word_counts = {}
    for token in query_tokens:
      if token in query_word_counts:
        query_word_counts[token]+=1
      else:
        query_word_counts[token] = 1
    scores = score_func(query_word_counts,index,idf)
    query_norm = 0
    for word in query_tokens:
      if word in idf:
        query_norm += (query_word_counts[word] * idf[word] )**2
    query_norm = math.sqrt(query_norm)
    solution = []
    for score in scores:
      cossim = scores[score]/(doc_norms[score] * query_norm)
      solution.append((cossim,score))
    solution.sort(key=lambda x:x[0], reverse = True)
    return solution

"""
Cosine Similarity Calculation Functions -- Finish
"""
"""
SVD -- Start
"""
with open(json_file_path) as f_2:
    documentss = [(x['product_name'], x['category'], x['about_product'])
                 for x in json.load(f_2)
                 if len(x['about_product'].split()) > 50]

#cosine similarity
#closest_words helper function -- cosine similarity
def closest_words(wti, word_in, words_representation_in, k = 10):
  index_to_word = {i:t for t,i in wti.items()}
  # print(word_in not in wti)
  if word_in not in wti: return "Not in vocab."
  #print("reached")
  #print("but reached here")

  sims = words_representation_in.dot(words_representation_in[wti[word_in],:])
  #print("reached")
  #print("but reached here")
  asort = np.argsort(-sims)[:k+1]
  #print("reached")
  #print([(index_to_word[i],sims[i]) for i in asort[1:]])
  return [(index_to_word[i],sims[i]) for i in asort[1:]]

def test_func():
  #  print(documentss[0][0])
  #  print(documentss[0][1])
  #  print(documentss[0][2])
  vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7,
                            min_df = 75)
  #print(type(vectorizer))
  #we're using the descriptions as our "documents"
  td_matrix = vectorizer.fit_transform([x[2] for x in documentss])
  # print(type(td_matrix))
  # print(td_matrix.shape)
  """
  output for shape is (1191,234) -> 1191 docs, 234 words in vocab
  """
  #run SVD
  docs_compressed, s, words_compressed = svds(td_matrix, k=40)
  words_compressed = words_compressed.transpose()
  # print(docs_compressed.shape)
  # print(s.shape)
  # #40 is size of latent dim, 234 in vocab
  # print(words_compressed.shape)

  #print(words_compressed)
  #doc to term representation
  word_to_index = vectorizer.vocabulary_
  #print(word_to_index.keys())
  #index_to_word = {i:t for t,i in word_to_index.items()}
  words_compressed_normed = normalize(words_compressed, axis = 1)

  word = 'charger'
  print("Using SVD:")
  try:
    for w, sim in closest_words(word_to_index, word, words_compressed_normed):
      try:
        print("{}, {:.3f}".format(w, sim))
      except:
        print("word not found")
    print()
  except:
     print("need better word")

  itw2 = {i:t for t,i in word_to_index.items()}

  for i in range(40):
    print("Top words in dimension", i)
    dimension_col = words_compressed[:,i].squeeze()
    asort = np.argsort(-dimension_col)
    print([itw2[i] for i in asort[:10]])
    print()

"""
SVD -- Finish
"""
"""
Helper Functions -- Start
"""
#this function takes in an integer and returns a category
def get_age_category(age):
   if age > 0 and age < 13:
      return "Child"
   elif age > 12 and age < 19:
      return "Teenager"
   elif age > 18 and age < 26:
      return "Young Adult"
   elif age > 25 and age < 61:
      return "Adult"
   else:
      return "Old"

#this function converts the database string repr of price into an integer
def price_to_int(price):
  res = ""
  try:
    index = 0
    while index < len(price):
       if price[index] == ".":
          break #round up
       elif price[index] == ",":
          index += 1
          continue
       else:
          res += price[index]
          index += 1
    
    #print(res)
    return int(res[1:]) + 1
    # period_index = price.find(".") #find where the period is
    # number = price[1:period_index] #take just the number
    # return int(number) + 1 #round up
  except:
    #print("runs")
    return 0 #just return small number -- if price is None

#general filter function called to filter original data according to filters
def filter(original_data,age = None,gender = None,pricing= None):
   filtered_data = []
   for item in data:
      item_price = item['actual_price']
      #print(price_to_int(item_price))
      if price_to_int(item_price) < int(pricing):
         #print(price_to_int(item_price))
         filtered_data.append(item)
   return filtered_data

#currently query is hardcoded 'puzzle creative fun' see the result in the terminal
#doesn't properly print results to the website
def json_search(query,age=None,gender=None,pricing=None):
    #first filter out products given age, gender, and pricing
    filtered_data = filter(data,age,gender,pricing)

    #filtered_data = data
    #run cosine similariity using query 
    dict_products = {}
    count = 1
    for i in filtered_data:
        dict_products[count] = tokenize(i['about_product'])
        count+=1

    #dict_products[1]
    
    inv_indx = build_inverted_index(dict_products)
    idf = compute_idf(inv_indx, len(filtered_data),
                  min_df=10,
                  max_df_ratio=0.1) 

    #inverted index for good words only 
    inv_idx = {key: val for key, val in inv_indx.items()
            if key in idf} 
    doc_norms = compute_doc_norms(inv_idx, idf, len(filtered_data))
    #query = 'Star Wars Han Solo'
    #runs cosine similarity
    results = index_search(query, inv_idx, idf, doc_norms)
    #print(results)
    doc_id_to_product = {}
    count = 1
    for i in filtered_data:
        doc_id_to_product[count] = i
        count+=1
    
    result_final =[]
    try:
      for i in range(10):
          result_final.append({'name': doc_id_to_product[results[i][1]]['product_name'], 'descr':doc_id_to_product[results[i][1]]['about_product'], 'url': doc_id_to_product[results[i][1]]['product_link']})
        #print(doc_id_to_product[results[i][1]]['product_name'])
      return json.dumps(result_final)

    except:
       return json.dumps({"error" : "not enough products"})
"""
Helper Functions -- Finish
"""   
    
@app.route("/")
def home():
    return render_template('base.html',title="sample html")

"""
Filters:
-age range
-gender
-pricing
"""

@app.route("/episodes", methods = ['POST'])
def episodes_search():
    test_func()

    request_data = request.json
    text = request_data["title"] #query -- ex. Star Wars Action Figure
    #optional filters -- ***if not used then pass in empty string "" ***
    age = request_data["age"] #number ex. 17 that we classify into child, teen, YA, adult, old
    """
    child: 0-13
    Teen: 13-18
    Young Adult: 18-25
    Adult: 25-60
    Old: 60+
    """
    gender = request_data["gender"] #either Male or Female
    pricing = request_data["pricing"] #limit of how much user wants to spend ex. 100

    # print(text)
    # print(type(pricing))
    #return json_search(text,pricing=pricing)
    return json.dumps({"message" : "hello"})

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=8000)