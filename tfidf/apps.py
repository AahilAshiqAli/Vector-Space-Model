from django.apps import AppConfig
from django.core.management import call_command
import os
from . import data_modules
tfidf_data = data_modules.tfidf_data
idf = data_modules.idf
document_list_df = data_modules.document_list_df

class TfidfConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tfidf'

    def ready(self):
        """
        Called when the Django app is ready for use.
        This is the place to include any startup code.
        """
        self.check_or_build_index()

    def check_or_build_index(self):
        from django.conf import settings
        import pandas as pd
        global tfidf_data,idf,document_list_df

        # Define the path to the index file
        index_file_path = os.path.join(settings.BASE_DIR, 'data', 'inverted_index.csv')
        document_list_path = os.path.join(settings.BASE_DIR, 'data', 'document_list.csv')

        # Check if the index file exists
        if os.path.exists(index_file_path) and os.path.exists(document_list_path):
            print("Loading existing inverted index and IDF values and document list from CSV files...")
            try:
                tfidf_data = {}
                idf = {}
                df = pd.read_csv(index_file_path)
                df['docid'] = pd.to_numeric(df['docid'])
                df['tf_in_doc'] = pd.to_numeric(df['tf_in_doc'])
                df['tf_idf'] = pd.to_numeric(df['tf_idf'])
                df['idf'] = pd.to_numeric(df['idf'])
                document_list_df = pd.read_csv(document_list_path)
                document_list_df['Total_Words'] = pd.to_numeric(document_list_df['Total_Words'])
                document_list_df.index = document_list_df.index.astype(int)
                grouped = df.groupby('term')
                for term, group in grouped:
                    tfidf_data[term] = group.drop(columns=['term','idf']).reset_index(drop=True)
                    idf[term] = group['idf'].unique()[0]
                data_modules.document_list_df = document_list_df
                data_modules.idf = idf
                data_modules.tfidf_data = tfidf_data
            except Exception as e:
                print(e)
                call_command('built-inverted-index')
        else:
            # If the file doesn't exist, run the management command to build it
            print("Inverted index file does not exist. Building a new one...")
            call_command('built-inverted-index')
            print('Printing from apps.py' ,document_list_df)
