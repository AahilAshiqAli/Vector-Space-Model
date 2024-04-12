from django.core.management.base import BaseCommand
import os
from django.conf import settings
import pandas as pd
import math
import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from ...apps import tfidf_data, idf
from ... import data_modules

document_list_df = data_modules.document_list_df
tfidf_data = data_modules.tfidf_data
idf = data_modules.idf

class Command(BaseCommand):
    help = 'Builds or loads the inverted index and IDF values'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting to build or load the inverted index and IDF values...")
        build_or_load_inverted_index()

def remove_punctuations_and_numbers(text):
    text_no_punctuations = re.sub(r'[^\w\s]', '', text)
    text_no_punctuations_numbers = re.sub(r'\d', '', text_no_punctuations)
    return text_no_punctuations_numbers

def build_or_load_inverted_index():
    # Define paths
    global document_list_df,tfidf_data,idf
    data_folder = os.path.join(settings.BASE_DIR, 'data')
    static_folder = os.path.join(data_folder, 'static')
    index_file_path = os.path.join(data_folder, 'inverted_index.csv')
    document_list_path = os.path.join(data_folder, 'document_list.csv')
    stopwords_file_path = os.path.join(data_folder, 'Stopword-List.txt')

    # Check if inverted index CSV file exists
    if os.path.exists(index_file_path):
        os.remove(index_file_path)
    if os.path.exists(document_list_path):
        os.remove(document_list_path)
        print("Existing index file removed.")

    with open(stopwords_file_path) as file:
        stop_words = word_tokenize(file.read())

    # Initialize variables
    index = {}
    document_list = {}
    total_docs = 0

    # Iterate over files in the static folder to make document list
    for filename in os.listdir(static_folder):
        file_path = os.path.join(static_folder, filename)
        if os.path.isfile(file_path):
            total_docs += 1
            with open(file_path, 'r',encoding='windows-1252') as file:
                word = file.read()
                text = remove_punctuations_and_numbers(word)
                words = word_tokenize(text)
                document_list[total_docs] = filename
                index[total_docs] = words
                        
    ps = PorterStemmer()
    modified_index = {} # index having docuemnt name and words in document
    for filename,tokens in index.items():
        modified_index[filename] = [ps.stem(token.lower()) for token in tokens if token not in string.ascii_letters and token not in stop_words]

    for filename,tokens in modified_index.items():
        modified_index[filename] = [len(tokens),tokens]
        document_list[filename] = (document_list[filename],len(tokens))
        
    inverted_index = {}
    # key = term, value = [doc frequency,{Docid : frequency}]

    for filename,tokens in modified_index.items():
        for token in tokens[1]:
            if token not in inverted_index.keys():
                inverted_index[token] = [1,{filename : 1}]
            else:
                dic = inverted_index[token]
                if filename in dic[1].keys():
                    dic[1][filename]+=1
                else:
                    dic[0] += 1
                    dic[1][filename] = 1     
                        
    def calculate_tfidf(idf, total_terms_in_doc, term_freq):
        tf = term_freq / total_terms_in_doc
        return tf * idf

    tfidf_data = {}  # Dictionary to store TF-IDF data for each term
    idf = {}
    for term, values in inverted_index.items():
        df = values[0]  # Document frequency is stored at index 0 of values
        idf[term] = math.log(total_docs / 1+df)
        tfidf_list = []  # List to store TF-IDF values for each document

        for docid, term_freq in values[1].items():
            total_terms_in_doc = document_list[docid][1]  # Total terms in the document
            tfidf = calculate_tfidf(idf[term], total_terms_in_doc, term_freq)
            tfidf_list.append({'docid': docid, 'tf_in_doc': term_freq, 'tf_idf': tfidf})

        # Convert tfidf_list into a DataFrame indexed by document IDs
        tfidf_df = pd.DataFrame(tfidf_list)
        tfidf_data[term] = tfidf_df
    # Save inverted index to CSV

    dataframes = []
    for term, df in tfidf_data.items():
        df['term'] = term  # Add a column for the term
        df['idf'] = idf[term] # Add a column for idf
        dataframes.append(df)
    tfidf_combined = pd.concat(dataframes)
    
    # Set the index to 'term' and save to a CSV file
    tfidf_combined.set_index('term').to_csv(index_file_path)
    document_list_df = pd.DataFrame(document_list.values(), index=document_list.keys(), columns=['Filename', 'Total_Words'])
    document_list_df.to_csv(document_list_path)
    data_modules.document_list_df = document_list_df
    print("Inverted index and IDF values built and saved to CSV files.")

