from flask import *
import os
from PyPDF2 import PdfReader
import re
import nltk
# nltk.download('punkt')
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.parsing.preprocessing import remove_stopwords
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

def pdf_extract(pdf_path):
    TEXT = ""
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        pages_count = len(pdf.pages)
        for i in range(pages_count):
            page = pdf.pages[i]
            TEXT = TEXT + page.extract_text()
    TEXT = re.sub(r'\s+', ' ', TEXT).strip()
    return TEXT

def clean_sentence(sentence, stopwords=False):
  sentence = sentence.lower().strip()
  sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
  if stopwords:
    sentence = remove_stopwords(sentence)
  return sentence

def getWordVec(word, model):
  sample = model['pc']
  vec = [0]*len(sample)
  try:
    vec = model[word]
  except:
    vec = [0]*len(sample)
  return (vec)

def Phrase_Embedding(phrase, embeddingmodel):
  sample = getWordVec('computer', embeddingmodel)
  vec = np.array([0]*len(sample))
  den = 0;
  for word in phrase.split():
    den = den+1
    vec = vec+np.array(getWordVec(word, embeddingmodel))
  return vec.reshape(1, -1)

def RetrieveAndPrint(question_embedding, sentence_embeddings, sentences):
  import sklearn
  from sklearn.metrics.pairwise import cosine_similarity
  max_sim = -1
  index_sim = -1
  for index, embedding in enumerate(sentence_embeddings):
    sim = cosine_similarity(embedding, question_embedding)[0][0]
    if sim > max_sim:
      max_sim = sim
      index_sim = index
  
  return index_sim
def get_cleaned_sentences(tokens, stopwords=False):
  cleaned_sentences = []
  for row in tokens:
    cleaned = clean_sentence(row, stopwords)
    cleaned_sentences.append(cleaned)
  return cleaned_sentences

glove_model = gensim.models.KeyedVectors.load('/Users/vanshgupta/Downloads/Flask QA System/glovemodel.mod') #REPLACE THIS PATH WITH YOUR PATH OF glovemodel.mod#
def glove_drive(pdf_path, question):
  pdf_txt = pdf_extract(pdf_path)

  tokens = nltk.sent_tokenize(pdf_txt)
  clean = get_cleaned_sentences(tokens, stopwords=True)
  cleaned_stopwords = get_cleaned_sentences(tokens, stopwords=False)
  sentences = cleaned_stopwords
  sentence_words = [[word for word in document.split()] for document in sentences]

  sentence_embeddings = []
  for sent in clean:
    sentence_embeddings.append(Phrase_Embedding(sent, glove_model))

  question_embedding = Phrase_Embedding(question, glove_model)
  index = RetrieveAndPrint(question_embedding, sentence_embeddings, cleaned_stopwords)
  return cleaned_stopwords[index]

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def upload_pdf():
  if request.method == 'POST':
    if(request.form.get('btn') == 'index'):
      upload = request.files['upload']
      upload.save(os.path.join('uploads',upload.filename))
      global pdf_path
      pdf_path = os.path.join('uploads',upload.filename)
      return redirect(url_for('QuesAns'))
    elif (request.form.get('btn') == 'QuesAns'):
      question = request.form.get('question')
      answer = glove_drive(pdf_path, question)
      return render_template('QuesAns.html', answer = answer, question = question)
  return render_template('upload.html')

@app.route('/QuesAns/', methods=['GET', 'POST'])
def QuesAns():
    return render_template('QuesAns.html')
     
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)