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
json_file_path = os.path.join(current_directory, 'results_categories.json')

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
    if text:
      text = text[0].lower()
      return re.findall(r'[a-z]+', text)
    else:
       return ""

def build_inverted_index(msgs:dict) -> dict:

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

def compute_idf(inv_idx, n_docs, min_df=2, max_df_ratio=0.95):
    solution = {}
    for word in inv_idx:
      if len(inv_idx[word]) >= min_df and (len(inv_idx[word])/n_docs) <= max_df_ratio:
        if word == 'barbie':
         print('here!')
        calc = n_docs/(1 + len(inv_idx[word]))
        calc = math.log2(calc)
        
       
        solution[word] = calc
    

    return solution

def compute_doc_norms(index, idf, n_docs):

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

    query_tokens = re.findall(r'[a-z]+', query.lower())
    #print(query_tokens)
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
    documentss = [(x['title'], x['main_category'], x['description'][0])
                 for x in json.load(f_2)
                 if len(x['description'][0].split()) > 50]

#cosine similarity
#closest_words helper function -- cosine similarity
def closest_words(wti, word_in, words_representation_in, k = 10):
  index_to_word = {i:t for t,i in wti.items()}
  print(word_in not in wti)
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

  # word = 'Makeup'
  # print("Using SVD:")
  # try:
  #   for w, sim in closest_words(word_to_index, word, words_compressed_normed):
  #     try:
  #       print("{}, {:.3f}".format(w, sim))
  #     except:
  #       print("word not found")
  #   print()
  # except:
  #    print("need better word")


  query = "Star Trek Toy stars space galaxy lasers"
  query_tfidf = vectorizer.transform([query]).toarray()
  #should be (1,vocab)
  query_tfidf.shape

  #words_compressed is shape (latent,vocab)
  query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

  docs_compressed_normed = normalize(docs_compressed)

  #number of res
  k = 5
  sims = docs_compressed_normed.dot(query_vec)
  asort = np.argsort(-sims)[:k+1]
  res = [(i, documentss[i][0],sims[i]) for i in asort[1:]]

  for i, proj, sim in res:
    print("({}, {}, {:.4f}".format(i, proj, sim))

  #itw2 = {i:t for t,i in word_to_index.items()}

  # for i in range(40):
  #   print("Top words in dimension", i)
  #   dimension_col = words_compressed[:,i].squeeze()
  #   asort = np.argsort(-dimension_col)
  #   print([itw2[i] for i in asort[:10]])
  #   print()

"""
SVD -- Finish
"""

"""
Functions for Query Filtering -- Start
"""
from nltk.stem import PorterStemmer
import re

#The below is for stemming
splitter = re.compile(r"""
    [.!?]       # split on punctuation
    """, re.VERBOSE)
query = "birthday gift for kids who like legos"
stemmer=PorterStemmer()
word_regex = re.compile(r"""
    (\w+)
    """, re.VERBOSE)
def getstems(sent):
    return [stemmer.stem(w.lower()) for w in word_regex.findall(sent)]

#these are for removing stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))
stop_words.add('gift') 
stop_words.add('present') 

"""
Functions for Stemming -- End
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
    return int(res) + 1
    # period_index = price.find(".") #find where the period is
    # number = price[1:period_index] #take just the number
    # return int(number) + 1 #round up
  except:
    #print("runs")
    return 0 #just return small number -- if price is None

#converts database float repr of rating to int (round down)
def average_rating_to_int(average_rating):
   try:
      return int(average_rating)
   except:
      return 0

#general filter function called to filter original data according to filters
def filter(original_data,age = None,gender = None,pricing= None):
   filtered_data = []
   for item in original_data:
      item_price = item['price']
      avg_rating = item['average_rating']
      #print(price_to_int(item_price))
      #print(average_rating_to_int(avg_rating))
      if (price_to_int(item_price) < int(pricing)) and (average_rating_to_int(avg_rating) > 2):
         #print(price_to_int(item_price))
         filtered_data.append(item)
   #print(filtered_data)
   return filtered_data

def filter_results_stars(results, doc_id_to_product):
   new_results = []
   for i in range(len(results)-1):
    try:
      
      #print(float(doc_id_to_product[results[i][1]]['average_rating']))
      #print(results[i][0])
     
      # print(abs(results[i][0]-results[i+1][0]))
      if (abs(results[i][0]-results[i+1][0]) < .01):
         curr_product = results[i][0] * .95 + float(doc_id_to_product[results[i][1]]['average_rating'])*.001*.04 + int(doc_id_to_product[results[i][1]]['rating_number'])*.001*.01
      else:
         curr_product = results[i][0]
      new_results.append((curr_product,results[i][1]))
      
    except:
       print('here')
       new_results.append((results[i][0],results[i][1]))
      #print('oops')
   new_results.sort(key=lambda x:x[0], reverse = True)
   return new_results


#currently query is hardcoded 'puzzle creative fun' see the result in the terminal
#doesn't properly print results to the website
def json_search(query,age=None,gender=None,pricing=None):
    #first filter out products given age, gender, and pricing
    filtered_data = filter(data,age,gender,pricing)

    #filtered_data = data
    #run cosine similariity using query 
    dict_products = {}
    count = 1

    #Perform stemming on the query
    sent_words_lower_stemmed = [getstems(sent) for sent in splitter.split(query)]

    allstemms=[w for sent in sent_words_lower_stemmed
              for w in sent]
    
    #remove stop words from the query
    final_query = []
    for w in allstemms:
        print(w)
        if not (w in stop_words):
            final_query.append(w)

    query = ""
    for w in final_query:
       query = query + w + " "
    
    print(type(filtered_data[0]['description'][0]))
    for i in filtered_data:
        #print(i['description'][0])
        
        #?
        curr_product = []
        for feature in i['features']:
           curr_product += tokenize(feature)
        dict_products[count] = curr_product+ tokenize(i['description'])
        count+=1

    #dict_products[1]
    
    inv_indx = build_inverted_index(dict_products)
    idf = compute_idf(inv_indx, len(filtered_data)) 

    #inverted index for good words only 
    inv_idx = {key: val for key, val in inv_indx.items()
            if key in idf} 
    doc_norms = compute_doc_norms(inv_idx, idf, len(filtered_data))

    #runs cosine similarity
    results = index_search(query, inv_idx, idf, doc_norms)
    #print(results)

    doc_id_to_product = {}
    count = 1
    for i in filtered_data:
        doc_id_to_product[count] = i
        count+=1
    result_final= []
    results = filter_results_stars(results,doc_id_to_product)

    try:
      for i in range(min(10,len(results))):          
        result_final.append({'name': doc_id_to_product[results[i][1]]['title'], 'price':doc_id_to_product[results[i][1]]['price'],'rating': doc_id_to_product[results[i][1]]['average_rating'], 'descr':doc_id_to_product[results[i][1]]['description'], 'url': "https://www.amazon.com/dp/" + doc_id_to_product[results[i][1]]['parent_asin']})
          #print(doc_id_to_product[results[i][1]]['product_name'])
      return json.dumps(result_final)
    except:
       return json.dumps({"error" : "something went wrong"})

    # try:
    #   for i in range(10):
    #       result_final.append({'name': doc_id_to_product[results[i][1]]['title'], 'descr':doc_id_to_product[results[i][1]]['description'], 'url': "https://www.amazon.com/dp/" + doc_id_to_product[results[i][1]]['parent_asin']})
    #     #print(doc_id_to_product[results[i][1]]['product_name'])
    #   return json.dumps(result_final)

    # except:
    #    return json.dumps({"error" : "not enough products"})
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
    return json_search(text,pricing=pricing)
    #return json.dumps({"message" : "hello"})

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=8000)