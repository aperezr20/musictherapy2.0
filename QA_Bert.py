
import os
import pandas as pd
from ast import literal_eval
from cdqa.utils.converters import pdf_converter  #converts to pandas dataframe
from cdqa.pipeline import QAPipeline     #Question Answer Pipeline
from cdqa.utils.download import download_model   #to download the pre-trained model
import sys
sys.path.insert(0, '/media/SSD0/aperezr20/AML/Language/mtpy/')
from pdf2txt import *


download_model(model='bert-squad_1.1', dir='./models')

#output = extract_text(outfile= 'test.txt',files=['/media/SSD0/aperezr20/AML/Language/mtpy/papers/Abromeit, 2003. The Newborn Individualized Developmental Care and Assessment Program (NIDCAP) as a Model for Clinical Music Therapy Interventions with Premature Infants.pdf'])

#breakpoint()
df=pdf_converter(directory_path='./papers')
df.head()


breakpoint()
cdqa_pipeline=QAPipeline(reader='./models/bert_qa.joblib',max_df=1.0)
cdqa_pipeline.fit_retriever(df=df)
query='How many full time employees are on Amazon roll?'
prediction=cdqa_pipeline.predict(query)
print('query:{}'.format(query))
print('answer:{}'.format(prediction[0]))
print('title:{}'.format(prediction[1]))
print('paragraph:{}'.format(prediction[2]))
