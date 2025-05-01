from django.shortcuts import render
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from django.conf import settings
import re
#from .apps import 
import pandas as pd
import numpy as np
import os
from . import data_modules
# Create your views here.

search_text = 'cancer overview'
doc_num = None
document_list_df = data_modules.document_list_df
tfidf_data = data_modules.tfidf_data
idf = data_modules.idf

#.......................................................................................................................
def calculate_cosine_similarity(query):
    global tfidf_data,idf,document_list_df
    ps = PorterStemmer()
    query_terms = [ps.stem(term) for term in query.split()]
    query_series = pd.Series(query_terms).value_counts()
    # Initialize an empty DataFrame to store TF-IDF scores for each document and term
    cosine = pd.DataFrame()
    query = {}
    # Iterate over each term in the query
    for term in set(query_terms):
        # Check if the term exists in the tfidf_data dictionary
        query[term] = idf[term] * query_series.loc[term]
        if term in tfidf_data.keys():
            # Retrieve the document IDs and TF-IDF scores for the term
            term_data = tfidf_data[term]['tf_idf'].to_frame()
            # Rename the 'tf_idf' column to the term name
            term_data.rename(columns={'tf_idf': term}, inplace=True)
            # Merge the term data with the existing DataFrame based on the 'docid'
        else:
            term_data = pd.DataFrame(columns=[term])
        cosine = pd.merge(left = cosine,right = term_data, how='outer', left_index=True, right_index=True)

    # Fill NaN values with 0
    cosine.fillna(0.0, inplace=True)
    # from here, calculating cosine similarity
    cosine_similarity = []

    # Compute magnitude (L2 norm) of the query vector
    query_magnitude = np.sqrt(np.sum(np.square(list(query.values()))))

    # Iterate over rows of the DataFrame
    for key, document_dict in cosine.iterrows():
        # Compute dot product of the query vector and the document vector
        dot_product = 0
        for term, tfidf_score in query.items():
            dot_product += tfidf_score * document_dict[term]  # Multiply corresponding TF-IDF scores

        # Compute magnitude (L2 norm) of the document vector
        document_magnitude = np.sqrt(np.sum(np.square(document_dict.values)))


        # Compute cosine similarity
        cosine_similarity.append((key,dot_product / (query_magnitude * document_magnitude)))

    # Sort cosine similarities in descending order
    cosine_similarity = [row for row in cosine_similarity if row[1] > 0.5]
    cosine_similarity.sort(reverse=True, key = lambda x : x[1])

    # Print the sorted cosine similarities
    print(cosine_similarity)
    return cosine_similarity



#........................................................................................................................
def index(request):
    global tfidf_data,idf,document_list_df
    return render(request,'index.html')

def about(request):
    global search_text
    if request.method == 'POST':
        search_text = request.POST.get('search_text', None)
    query_result = calculate_cosine_similarity(search_text)
    docs_list = [os.path.join(settings.BASE_DIR, 'data', 'static',document_list_df.loc[row[0],'Filename']) for row in query_result]
    read = []
    length = len(query_result)
    for file_name in docs_list:
      with open(file_name, 'r',encoding='windows-1252') as file:
        word = file.read()
        read.append(word[:300]) 
      
    qw = []   
    for i in range(length):
      q = {"res" : document_list_df.loc[query_result[i][0],'Filename'], "score" : round(query_result[i][1],3) ,  "name" : docs_list[i], "doc_info" : read[i]}
      qw.append(q)
   
            
    return render(request,'about.html',{'q' : qw, "frequency" : length})

def services(request):
    global doc_num
    try:
        if request.method == 'GET':
            doc_num = request.GET.get('document_number')
        print(doc_num)
        with open(doc_num) as file:
            detail = file.read()
    except:
        return render(request, 'question.html', {'error_message': 'Select document first from Search results. Previous page not saved'})
    return render(request,'services.html',{'d' : detail})