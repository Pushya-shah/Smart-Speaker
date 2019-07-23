import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tabulate import tabulate
import speech_recognition as sr
import os
from time import sleep

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" 
embed = hub.Module(module_url)


r=sr.Recognizer()


similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_sentences_encodings = embed(similarity_input_placeholder)

qa = {
    
      'easy': {'What is the main difference between list and tuple?':'The difference between list and tuple is that list is mutable/editable while tuple is not.',
              'Is python case sensitive?':'yes',
              'What is the difference between Python Arrays and lists?':'arrays can hold only a single data type elements whereas lists can hold any data type elements.',
              'What is a dictionary in Python?':'It defines one-to-one relationship between keys and values. Dictionaries contain pair of keys and their corresponding values. Dictionaries are indexed by keys.'
              },
      
      'normal': {"What are the datatypes in python?":"integer, float, strings, set, list, tuple, dictionary",
                'What is flask?':'Flask is a micro framework primarily build for a small application with simpler requirements',
                'Define encapsulation in Python?' : 'Encapsulation means binding the code and the data together. A Python class in an example of encapsulation.' 
                },
      
      'hard': {"What is a lambda function?":"An anonymous function is known as a lambda function. This function can have any number of parameters but, can have just one statement.",
              'Django is based on which design pattern?':'Django closely follows the MVC (Model View Controller) design pattern.',
              'What is the difference between supervised and unsupervised machine learning?':'Supervised learning requires training labeled data while unsupervised learning doesn\'t requires labeled data'
              }
     
     }


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())

    result = 0

    for q_level, q_info in qa.items():
        print("\nQuestion's level is ", q_level)

        for ques in q_info:
            sentences = []
            os.system("say {}".format(ques))
            print(ques)
            actual_ans = q_info[ques]
            sleep(2)
            os.system("arecord --duration=10 smart.wav")

            harvard = sr.AudioFile('smart.wav')
            with harvard as source:
              r.adjust_for_ambient_noise(source)
              audio = r.record(source)

            input_ans=r.recognize_google(audio)
            print(input_ans)

            sentences.append(actual_ans)
            sentences.append(input_ans)
            sentences_embeddings = session.run(similarity_sentences_encodings, feed_dict={similarity_input_placeholder: sentences})
            simm = np.inner(sentences_embeddings[0], sentences_embeddings[1])
            result = result+simm

    print(result)
