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
json_file_path = os.path.join(current_directory, 'results.json')

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

"""
Making data with categories -- Start
"""
data_with_categories = {}
for d in data:
   if d["main_category"] in data_with_categories:
      data_with_categories[d["main_category"]].append(d)
   else:
    data_with_categories[d["main_category"]]= [d]

"""
Making data with categories 
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
    print("doc norms")
    print(len(solution))
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
    #print(scores)
    for item_id in list(scores.keys()):
      #print(item_id)
      denom = doc_norms[item_id-1] * query_norm
      if denom != 0:
        cossim = scores[item_id]/denom
        solution.append((cossim,item_id))
    solution.sort(key=lambda x:x[0], reverse = True)
    return solution

"""
Cosine Similarity Calculation Functions -- Finish
"""
"""
SVD -- Start
"""
with open(json_file_path) as f_2:
    #only take docs whose description is more than 50 words
    # documentss = [(x['title'], x['main_category'], x['description'][0], x['price'])
    #              for x in json.load(f_2)
    #              if len(x['description'][0].split()) > 50]
    documentss = []
    for x in json.load(f_2):
       to_add = ""
       for d in x['description']:
          to_add += d
       for feature in x['features']:
          to_add += feature
       documentss.append((x['title'], x['main_category'], to_add, x['price'], x['average_rating'],x['parent_asin']))
    print('len documents')
    print(len(documentss))
       
    # ocumentss = [(x['title'], x['main_category'], x['description'][0], x['price'])
    #              for x in json.load(f_2)
    #              if len(x['description'][0].split()) > 50]
    
    

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

"""
Prints the categories of our data
"""
def get_categories(data):
  d1 = {} 
  for i in data:
    curr_category = i[1] #main category
    if curr_category in d1:
       d1[curr_category] += 1
    else:
       d1[curr_category] = 1

  for i in list(d1.keys()):
     print(str(i) + " : " + str(d1[i]))

  """
  This are the main categories of our dataset:
  All Beauty : 247
  AMAZON FASHION : 320
  Amazon Home : 404
  Arts, Crafts & Sewing : 162
  Tools & Home Improvement : 29
  Health & Personal Care : 148
  Office Products : 117
  Toys & Games : 518
  Sports Collectibles : 6
  Collectible Coins : 2
  Industrial & Scientific : 30
  None : 53
  Baby : 106
  Collectibles & Fine Art : 2
  Cell Phones & Accessories : 16
  Sports & Outdoors : 209
  Grocery : 5
  All Electronics : 61
  Pet Supplies : 8
  Musical Instruments : 5
  Digital Music : 120
  Movies & TV : 4
  Computers : 90
  Camera & Photo : 32
  Home Audio & Theater : 26
  Car Electronics : 8
  GPS & Navigation : 2
  Portable Audio & Accessories : 1
  Handmade : 1
  Automotive : 5
  Video Games : 113
  Software : 2
  """
  
"""
Prints the singular values from svd
"""
def print_singular_values(sigma):
   for i in sigma:
      print(i)
   
from sklearn.feature_extraction import text

def first_svd(query):
  get_categories(documentss)
  print(documentss[0][0])
  print(documentss[0][1])
  print(documentss[0][2])
  my_stop_words = text.ENGLISH_STOP_WORDS.union(["person","like"])
  vectorizer = TfidfVectorizer(stop_words = ["person", "like"], max_df = .95,
                            min_df = 10)
  
  #print(vectorizer)
  #we're using the descriptions as our "documents"
  td_matrix = vectorizer.fit_transform([x[2] for x in documentss])
  # print(type(td_matrix))
  print(td_matrix.shape)
  """
  output for shape is (2852,340) -> 2852 docs, 340 words in vocab
  """
  #run SVD
  k = 10
  docs_compressed, s, words_compressed = svds(td_matrix, k)
  #print_singular_values(s)
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


  #what if query vocab does not appear in original vocab? -- .transform() just ignore thems
  
  query = query.lower()

  #checks which terms in query are in vocab/check for query strength
  query_strength = 0
  total = 0
  q_words = query.split()
  print(q_words)
  for i in q_words:
    if i in word_to_index:
        query_strength += 1
        print(i + " is in vocab")
    total += 1

  query_strength /= total
  print("query strength " + str(query_strength))
  
  """
  each entry in query_tfidf is from 0-1, describes the tfidf weight of the term
  if the tdidf weight is close 1, that means the word appears alot in the query
  AND it doesn't appear that frequently in other docs (ex. "the")
  """
  query_tfidf = vectorizer.transform([query]).toarray()
  #should be (1,vocab) -- weights
  print("query tfidf")
  # print(query_tfidf) 
  print(query_tfidf.shape)

  #words_compressed is shape (latent,vocab)
  """
  words compressed is each word expressed in terms of latent dimensions (vocab, latent dim)
  query_tfidf is as described above, with shape (1,vocab)

  we perform matrix multiplication (using np.dot) to "project the representation of our query onto latent dim

  the end result is (10,), in other words an array of length latent dimension
  each value is how much the query matches up with latent dimension

  values are between [-1,1]

  positive values mean more relevant it is to that dim
  ex. if the dim was sports, query would have words like soccer basketball

  negative values mean more irrelevant it is to that dim
  ex. if the dim was sports, query would have words like robot, alien

  higher magnitude values ex. 0.5 vs 0.9 mean that it has more of an influence
  """
  query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()
  #query vec 
  print("query_vec")
  print(query_vec)
  print(query_vec.shape)

  docs_compressed_normed = normalize(docs_compressed)

  #number of res
  k = 5
  sims = docs_compressed_normed.dot(query_vec) #highest overlap in terms of latent dim
  asort = np.argsort(-sims)[:k+1]
  res = [(i, documentss[i][0],documentss[i][1], documentss[i][2], documentss[i][3], documentss[i][4], documentss[i][5],sims[i]) for i in asort[1:]]

  most_freq_cat = {}
  result = []
  for i, proj, cat ,descr,price,rating,url, sim in res:
    #documentss.append((x['title'], x['main_category'], to_add, x['price'], x['average_rating'],x['parent_asin']))
    print("({}, {}, {}, {:.4f}".format(i, proj,cat, sim))
    #this is to match the result of json_search
    result.append({'name': proj, 'price':price,'rating': rating, 'descr':descr, 'url': "https://www.amazon.com/dp/" + url})
    #result.append((sim,i))
  return result
  #   if cat in most_freq_cat:
  #      most_freq_cat[cat] += 1
  #   else:
  #      most_freq_cat[cat] = 1
  
  # most_freq_cat_str = ""
  # max = 0
  # for i in list(most_freq_cat.keys()):
  #    if most_freq_cat[i] > max:
  #       max = most_freq_cat[i]
  #       most_freq_cat_str = i
     
  # print("most freq category: ")
  # print(most_freq_cat_str)
  # #next section displays the words that are expressed the most in latent dimensions
  # itw2 = {i:t for t,i in word_to_index.items()}

  # for i in range(10):
  #   print("Top words in dimension", i)
  #   dimension_col = words_compressed[:,i].squeeze()
  #   asort = np.argsort(-dimension_col)
  #   print([itw2[i] for i in asort[:10]])
  #   print()

  #   return most_freq_cat_str
  
def improved_svd(query,category):
  if category == None:
     return []
  #category = first_svd(query)
  # print("Category: " + category)
  #filter out documents by the chosen category
  new_documents = []
  for i in documentss:
     if i[1] == category:
        #print(i[1])
        new_documents.append(i)

  #restrictions relaxed a little
  vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7,
                            min_df = 25)
  
  #print(vectorizer)
  #we're using the descriptions as our "documents"
  td_matrix = vectorizer.fit_transform([x[2] for x in new_documents])
  # print(type(td_matrix))
  print(td_matrix.shape)
  """
  output for shape is (2852,340) -> 2852 docs, 340 words in vocab
  """
  #run SVD
  k = 10
  docs_compressed, s, words_compressed = svds(td_matrix, k)
  #print_singular_values(s)
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


  #what if query vocab does not appear in original vocab? -- .transform() just ignore thems
  
  query = query.lower()

  #checks which terms in query are in vocab/check for query strength
  query_strength = 0
  total = 0
  q_words = query.split()
  print(q_words)
  for i in q_words:
    if i in word_to_index:
        query_strength += 1
        print(i + " is in vocab")
    total += 1

  query_strength /= total
  print("query strength " + str(query_strength))
  
  """
  each entry in query_tfidf is from 0-1, describes the tfidf weight of the term
  if the tdidf weight is close 1, that means the word appears alot in the query
  AND it doesn't appear that frequently in other docs (ex. "the")
  """
  query_tfidf = vectorizer.transform([query]).toarray()
  #should be (1,vocab) -- weights
  print("query tfidf")
  # print(query_tfidf) 
  print(query_tfidf.shape)

  #words_compressed is shape (latent,vocab)
  """
  words compressed is each word expressed in terms of latent dimensions (vocab, latent dim)
  query_tfidf is as described above, with shape (1,vocab)

  we perform matrix multiplication (using np.dot) to "project the representation of our query onto latent dim

  the end result is (10,), in other words an array of length latent dimension
  each value is how much the query matches up with latent dimension

  values are between [-1,1]

  positive values mean more relevant it is to that dim
  ex. if the dim was sports, query would have words like soccer basketball

  negative values mean more irrelevant it is to that dim
  ex. if the dim was sports, query would have words like robot, alien

  higher magnitude values ex. 0.5 vs 0.9 mean that it has more of an influence
  """
  query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()
  #query vec 
  print("query_vec")
  print(query_vec)
  print(query_vec.shape)

  docs_compressed_normed = normalize(docs_compressed)

  #number of res
  res = []
  k = 6
  sims = docs_compressed_normed.dot(query_vec) #highest overlap in terms of latent dim
  asort = np.argsort(-sims)[:k+1]

  # index = 0
  # while len(res) < 3 or index < len(asort):
  #   i = asort[1:]
  #   if new_documents[i][3] < pricing:
  #      res.append((i, new_documents[i][0],new_documents[i][1],sims[i]))

  #   index += 1
     
  res = [(i, documentss[i][0],documentss[i][1], documentss[i][2], documentss[i][3], documentss[i][4], documentss[i][5],sims[i]) for i in asort[1:]]

  #most_freq_cat = {}
  result = []
  for i, proj, cat ,descr,price,rating,url, sim in res:
    #documentss.append((x['title'], x['main_category'], to_add, x['price'], x['average_rating'],x['parent_asin']))
    print("({}, {}, {}, {:.4f}".format(i, proj,cat, sim))
    #this is to match the result of json_search
    result.append({'name': proj, 'price':price,'rating': rating, 'descr':descr, 'url': "https://www.amazon.com/dp/" + url})
    #result.append((sim,i))
  return result



   
"""
Improve SVD by categorization
"""

"""
SVD -- Finish
"""

"""
Functions for Query Filtering -- Start
"""
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

#The below is for stemming
splitter = re.compile(r"""
    [.!?]       # split on punctuation
    """, re.VERBOSE)
#query = "birthday gift for kids who like legos"
stemmer=PorterStemmer()
#wordnetlemmatizer inspired by https://stackoverflow.com/questions/24517722/how-to-stop-nltk-stemmer-from-removing-the-trailing-e
wnl = WordNetLemmatizer()
word_regex = re.compile(r"""
    (\w+)
    """, re.VERBOSE)

def getstems(sent):
    return [(wnl.lemmatize(w.lower()) if wnl.lemmatize(w.lower()).endswith('e') else stemmer.stem(w.lower())) for w in word_regex.findall(sent)]

#these are for removing stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))
stop_words.add('gift') 
stop_words.add('present') 
stop_words.add('person') 
stop_words.add('like') 

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
def filter(original_data,age = None,gender = None,pricing= None,category= None):
   filtered_data = []
   for item in original_data:
      item_price = item['price']
      #avg_rating = item['average_rating']
      m_category = item['main_category']
      #print(price_to_int(item_price))
      #print(average_rating_to_int(avg_rating))
      if (price_to_int(item_price) < int(pricing))  and (m_category == category or category == "all") :
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
def json_search(query,age=None,gender=None,pricing=None,category=None):
    #first filter out products given age, gender, and pricing
    if category == None:
       filtered_data = data
    else:
      filtered_data = filter(data_with_categories[category],age,gender,pricing,category)

    #print(len(filtered_data))
    #filtered_data = data
    #run cosine similariity using query 
    dict_products = {}
    count = 1
    
    
    #print(type(filtered_data[0]['description'][0]))
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
    if category == None:
       results_svd = first_svd(query)
    else:
      results_svd = improved_svd(query,category)
    results = results[:10] 
    try:
      for i in range(min(16,len(results))):          
        result_final.append({'name': doc_id_to_product[results[i][1]]['title'], 'price':doc_id_to_product[results[i][1]]['price'],'rating': doc_id_to_product[results[i][1]]['average_rating'], 'descr':doc_id_to_product[results[i][1]]['description'], 'url': "https://www.amazon.com/dp/" + doc_id_to_product[results[i][1]]['parent_asin']})
        # print(doc_id_to_product[results[i][1]]['title'])
        # print(results[i])
      result_final = result_final + results_svd
      print(result_final)
      print('here')
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

@app.route("/not-helpful", methods=['POST'])
def receive_not_helpful():
    data = request.get_json()
    return jsonify({"status": "success", "message": "Data received"})

@app.route("/episodes", methods = ['POST'])
def episodes_search():
    #test_func("Skincare cleanser for girl with oily skin")

    request_data = request.json
    text = request_data["title"] #query -- ex. Star Wars Action Figure
    #optional filters -- ***if not used then pass in empty string "" ***
    #age = request_data["age"] #number ex. 17 that we classify into child, teen, YA, adult, old
    """
    child: 0-13
    Teen: 13-18
    Young Adult: 18-25
    Adult: 25-60
    Old: 60+
    """
    #gender = request_data["gender"] #either Male or Female
    pricing = request_data["pricing"] #limit of how much user wants to spend ex. 100

    # print(text)
    # print(type(pricing))
    #improved_svd("Cleanser for girl with oily skin",pricing)

    """
    best_cat either equals "all" or the category thats determined by svd 
    changed by commenting out one or the other
    """
    sent_words_lower_stemmed = [getstems(sent) for sent in splitter.split(text)]

    allstemms=[w for sent in sent_words_lower_stemmed
              for w in sent]
              
    final_query = []
    for w in allstemms:
        print(w)
        if not (w in stop_words):
            final_query.append(w)

    query = ""
    for w in final_query:
       query = query + w + " "

  
    #TODO: this field should be an input from the UI, it is ok to be None
    best_cat = "Books"
    return json_search(query,pricing=pricing,category=None)
    #return json.dumps({"message" : "hello"})

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=8000)

#test