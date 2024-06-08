# ARIES-PDF-ANSWERING-AI

PROJECT OVERVIEW:
The project focuses on creating a simple interface which allows the user to upload a PDF and ask questions relevant to that PDF. The project utilises Natural Language Processing (NLP) and Global Vectors (GloVe) techniques to generate "score" of relevancy from the question and the output among possible outputs with the highest score is given as answer.

Installation Instructions:
The installation instructions are simple, save all the code files in a directory, make sure all the libraries and dependencies are installed along with glovemodel.mod file. Run the interface.py flask application on the local host server to achieve working of the application.

Usage:
The usage involves choosing a PDF from which we want to ask question, uploading it and then asking relevant question to get the answer.

Dependencies:
numpy
pandas 
flask
nltk
re
gensim
PyPDF2
sklearn
