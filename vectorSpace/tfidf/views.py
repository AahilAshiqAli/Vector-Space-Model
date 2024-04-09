from django.shortcuts import render, HttpResponse
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from django.conf import settings
import os
import re
# Create your views here.

filename_dictionary = ['1.txt', '2.txt', '3.txt', '7.txt', '8.txt', '9.txt', '11.txt', '12.txt', '13.txt', '14.txt', '15.txt', '16.txt', '17.txt', '18.txt', '21.txt', '22.txt', '23.txt', '24.txt', '25.txt', '26.txt']

#.......................................................................................................................


def remove_punctuations_and_numbers(text):
  
    text_no_punctuations = re.sub(r'[^\w\s]', '', text)

    text_no_punctuations_numbers = re.sub(r'\d', '', text_no_punctuations)

    return text_no_punctuations_numbers




#........................................................................................................................


