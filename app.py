#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify, render_template
import os


# In[2]:


import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
from spacy.lang.en import English


# In[3]:


loaded_model = tf.keras.models.load_model("skimlit_tribrid_model/")


# In[4]:


app = Flask(__name__, template_folder='templates')


# In[5]:


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


# In[6]:


def split_chars(text):
    return " ".join(list(text))

def predictor(data):
    label_encoder = LabelEncoder()
    labels = ['OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS','BACKGROUND']
    labels = label_encoder.fit_transform(labels)
    abstract1 = data
    nlp = English() 
    sentencizer = nlp.create_pipe("sentencizer") 
    nlp.add_pipe('sentencizer') 
    doc = nlp(abstract1) 
    abstract_lines = [str(sent) for sent in list(doc.sents)]
    total_lines_in_sample = len(abstract_lines)
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15) 
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]
    test_abstract_pred_probs = loaded_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                   test_abstract_total_lines_one_hot,
                                                   tf.constant(abstract_lines),
                                                   tf.constant(abstract_chars)))
    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
    test_abstract_pred_classes = [label_encoder.classes_[i] for i in test_abstract_preds]
    o = ''
    m = ''
    r = ''
    c = ''
    b = ''
    for i, line in enumerate(abstract_lines):
        if(test_abstract_pred_classes[i]) == 'OBJECTIVE':
            o = o + line + ' '
        elif(test_abstract_pred_classes[i]) == 'METHODS':
            m = m + line + ' '
        elif(test_abstract_pred_classes[i]) == 'RESULTS':
            r = r + line + ' '
        elif(test_abstract_pred_classes[i]) == 'CONCLUSIONS':
            c = c + line + ' '
        elif(test_abstract_pred_classes[i]) == 'BACKGROUND':
            b = b + line + ' '
    if(len(o)) > 0:
        o = 'OBJECTIVE: '+ o
    if(len(m)) > 0:
        m='METHODS: '+m
    if(len(r)) > 0:
        r='RESULTS: '+r
    if(len(c)) > 0:
        c='CONCLUSIONS: '+c
    if(len(b)) > 0:
        b='BACKGROUND: '+b 
    return (o,m,r,c,b)


# In[7]:


@app.route('/result', methods = ['POST', 'GET'])
def result():
    string2 = request.form["paragraph_text"]
    prediction = predictor(string2)
    return render_template('result.html',prediction = prediction)   


# In[ ]:


if __name__ == '__main__':
    app.run()



