from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
#models
from .models import Profile

from tensorflow.keras.applications.inception_v3 import InceptionV3

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from time import time
from os import listdir
from keras.models import load_model

from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

import pickle

def load_doc(filename):
    with open(filename) as file:
        text = file.read()
        return text
filename = 'C:/Users/97798/Desktop/Practise/New folder/Flickr_TextData/Flickr8k.token.txt'
doc = load_doc(filename)

def load_set(path):   
    # Loading the file containing the list of photo identifier
    file = load_doc(path)    
    # Creating a list for storing the identifiers
    images = list()    
    # Traversing the file one line at a time
    for line in file.split('\n'):
        if len(line) < 1:
            continue       
        # Image name contains the extension as well but we need just the name
        identifier = line.split('.')[0]      
        # Adding it to the list of photos
        images.append(identifier)
        
    # Returning the set of photos created
    return set(images)

path = 'C:/Users/97798/Desktop/Practise/New folder/Flickr_TextData/Flickr_8k.trainImages.txt'

train = load_set(path)

def load_cleaned_captions(path, images):
    file = load_doc(path)
    captions = {}         
    for i in file.split('\n'): #i =line, \n = new line
        # splitting the line at white spaces
        words = i.split()
        img_id = words[0]
        img_caption =  words[1:]
        if img_id in images:
            #creating list of captions if needed
            if img_id not in captions:
                captions[img_id] = list()
                cap = 'startofseq ' + ' '.join(img_caption) + ' endofseq'
            captions[img_id].append(cap)
            
    return captions

train_captions = load_cleaned_captions('captions.txt', train)

# Create a list of all the training captions
all_train_captions = []
for captions in train_captions.values():
    for caption in captions:
        all_train_captions.append(caption)

count_words = {}
for k in all_train_captions:
    for word in k.split():
        if word not in count_words:
            count_words[word] = 0

        else:
            count_words[word] += 1
            

THRESH = 10
count = 1
vocab = {}
for k,v in count_words.items():
    if count_words[k] > THRESH:
        vocab[k] = count
        count += 1
        

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1 # one for appended 0's


def extraction_features(filename):
    # load the model
    model = InceptionV3()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
    image = load_img(filename, target_size=(299, 299))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape(1,299,299,3)
    # prepare the image for the Inception model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image)
    return feature



def greedySearch(model,photo,MAX_LEN):	
    in_text = 'startofseq'
    for i in range(MAX_LEN):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=MAX_LEN)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endofseq':
            break
    return in_text


# Create your views here.
def index(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        image = fs.save(uploaded_file.name,uploaded_file)
        image_path = "C:/Users/97798/Desktop/Explaination/imagecaption/media/"+image
        extracted_image = extraction_features(image_path)
        max_length = 30
        model = load_model('C:/Users/97798/Desktop/Explaination/imagecaption/model-rms.h5')
        generated_caption = greedySearch(model, extracted_image, max_length)
        stopwords = ['startofseq','endofseq']
        newWords = generated_caption.split()
        resultwords = [word for word in newWords if word.lower() not in stopwords]
        result = ' '.join(resultwords)

        context['result'] = result
        context['image_name'] = uploaded_file.name
        context['url'] = fs.url(image)

    context['datas'] = Profile.objects.all()

    return render(request,"home.html",context)

from .form import ImageForm
from django.contrib.auth.mixins import LoginRequiredMixin

def home(request):
    if request.method == 'POST':
        form = ImageForm(data=request.POST,files=request.FILES)
        if form.is_valid():
            form.instance.author = request.user
            form.save()
            return render(request,"index.html")     
    else:
        form = ImageForm()   
    img = Profile.objects.all()
    return render(request,"index.html",{"img":img,"form":form})
   

