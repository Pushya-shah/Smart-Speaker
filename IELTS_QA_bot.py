qa = {

'Are international superstars popular in your country?':'Yes, nowadays they are becoming very popular because of the music festivals organized in our country. Also, the people have access to their music via TV channels or websites like Youtube.',
'Do you want to be a superstar?':'No, I would not like to be a superstar because I do not like the attention they get at public places. Moreover, superstars always have to appear good and there is a lot of pressure in their lives.',
'How do you feel when you see people throw garbage on the street?':'I feel bad when I see people throwing garbage on the road. Few times I have told them to pick it up and dispose it off properly',
'Do you like to see the sky?':'Yes I love to see clear blue sky and appreciate the nature. Looking at the sky reminds me that there are no limits and boundaries in life and we can achieve anything we want.',     
'Which is a good place to see the stars?':'I think mountains are the best place to see the stars. There is less pollution in the mountains so the stars are clearly visible and they also appear more closer.',     
'Did your parents teach you to share when you were a child?':'Yes my parents taught me that sharing is caring. They always told me to share my toys with my friends.',
'Do you think we should drink a lot of water?':'Yes we should definitely drink a lot of water as it is good for our health. Drinking water has lot of health benefits, it keeps our skin and body hydrated. If we do not drink enough water, our body will suffer from dehydration.',
'Do you think parents should teach children to save money.':'Yes, as I already mentioned parents should teach children about money management as it is an important skill in life',
'Do you think children should play sports regularly?':'Yes children should regularly play sports at it helps them in staying active and healthy. It is also a good break from their monotonous study routine.',
'Do you often read newspapers?':'Yes, I often read newspapers. I read the newspaper every day. I come to know what is happening around me and in the world.'

}

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import speech_recognition as sr
import os
from time import sleep
module_url = "/tmp/tfhub_modules/96e8f1d3d4d90ce86b2db128249eb8143a91db73" 
embed = hub.Module(module_url)
r=sr.Recognizer()

similarity_input_placeholder = tf.compat.v1.placeholder(tf.string, shape=(None))
similarity_sentences_encodings = embed(similarity_input_placeholder)
result=0
with  tf.compat.v1.Session() as session:
	session.run(tf.compat.v1.global_variables_initializer())
	session.run(tf.compat.v1.tables_initializer())
	for ques in qa:
	# print(ques)
		os.system("say {}".format(ques))
		orig_ans=qa[ques]
		# print("hello")
		sleep(5)
		os.system("arecord --duration=5 smart.wav")
		# print("hi")
		harvard = sr.AudioFile('smart.wav')
		with harvard as source:
			r.adjust_for_ambient_noise(source)
			audio = r.record(source)
		# ans=r.recognize_google(audio)
		# print("i am here")
		input_ans=r.recognize_google(audio)
		print(input_ans)
		sentences=[orig_ans,input_ans]
		sentences_embeddings = session.run(similarity_sentences_encodings, feed_dict={similarity_input_placeholder: sentences})
		similarity = np.inner(sentences_embeddings[0], sentences_embeddings[1])
		result=result+similarity
print(result*10)   