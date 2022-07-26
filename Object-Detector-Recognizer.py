# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:47:22 2021

@author: Youssef
"""

import cv2
import os
import numpy as np
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.applications.vgg16 import VGG16

        
data_path = r"E:\University\3-Third Year\Artificial intelligence\Project\Dataset II\BasicTraining\train"
img_size=256             
counter=0            
X=[]
Y=[]

categories=os.listdir(data_path)

for category in categories:                                                            # this loop to know how many images in categories
    folder_path=os.path.join(data_path,category)                                       # make folder empty has the same path for dataset
    img_names=os.listdir(folder_path)                                                  # put each image in this folder
    
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        fullpath=os.path.join(data_path,category,img_name)
        try:
            img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size,img_size))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            X.append(img)
            Y.append(category)
            counter+=1
            print("Reprocessing Image Number: ",counter)
        except:
            print("Error in ==> ",counter)

imgs=np.array(X)
lbls=np.array(Y)
del X
del Y

#Label Encoding
le = preprocessing.LabelEncoder()
le.fit(lbls)
lbls_encoded = le.transform(lbls)

#Train and Test Split
train_x, test_x,train_y, test_y = train_test_split(imgs,lbls_encoded,test_size=0.1)

#Normalization
train_x, test_x = train_x / 255.0,  test_x / 255.0

#One Hot Encoding
y_train_one_hot = to_categorical(train_y)
y_test_one_hot = to_categorical(test_y)

#Feature Extraction
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size,img_size, 3))

for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  
feature_extractor=VGG_model.predict(train_x)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_RF_DT = features

#Random Forest Training

RF_model = RandomForestClassifier(n_estimators = 100, random_state = 0)

RF_model.fit(X_for_RF_DT, train_y)


#Decision Tree Training (Has Less Accuracy)


from sklearn.tree import DecisionTreeClassifier

DT_model= DecisionTreeClassifier(criterion='gini',max_depth=200,random_state = 0) 

DT_model.fit(X_for_RF_DT, train_y)


#Testing
###########

#Feature Extraction For Test Data
feature_extractor_test=VGG_model.predict(test_x)
features_test = feature_extractor_test.reshape(feature_extractor_test.shape[0], -1)

#Prediction using trained RF
prediction_RF = RF_model.predict(features_test)
prediction_RF_Normal = le.inverse_transform(prediction_RF)



#Prediction using trained DT (Has Less Accuracy)
prediction_DT = DT_model.predict(features_test)
prediction_DT_Normal = le.inverse_transform(prediction_DT)



test_y_Normal = le.inverse_transform(test_y)
print ("Accuracy Using Random Forest = ", metrics.accuracy_score(test_y_Normal, prediction_RF_Normal)*100,"%")
print ("Accuracy Using Decision Trees = ", metrics.accuracy_score(test_y_Normal, prediction_DT_Normal)*100,"%")

##### GUI Part #####




from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
 
root = tk.Tk()
root.geometry("1500x1050")  # Size of the window 
root.resizable(width=False, height=False)
root.title('Object Detector')
root['background']='#222227' 
my_font1=('times', 18, 'bold')
my_font2=('times', 12, 'bold')
label = tk.Label(root,text='Upload Files & Detect',width=30,font=my_font1)
label.grid(row=1,column=1)
label.place(anchor = CENTER, relx = .5, rely = .025)
 

b1 = tk.Button(root, text='Upload Images', 
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1,pady=5)
b1.place(anchor = CENTER, relx = .5, rely = .070)
def upload_file():
    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png'),('Jpeg Files', '*.jpeg')]   # types of files to select 
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
    col=1 # start from column 1
    row=3 # start from row 3 
    for pathgui in filename:
        img=Image.open(pathgui)# read the image file
        list_of_images = []
        img_preprocessed = cv2.imread(pathgui, cv2.IMREAD_COLOR)
        img_preprocessed = cv2.resize(img_preprocessed, (img_size,img_size))
        img_preprocessed = cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2BGR)
        list_of_images.append(img_preprocessed)
        arr = np.array(list_of_images)
        feature_extractor_input=VGG_model.predict(arr)
        features_input = feature_extractor_input.reshape(feature_extractor_input.shape[0], -1)
        prediction_RF_input = RF_model.predict(features_input)
        prediction_RF_input_Normal = le.inverse_transform(prediction_RF_input)
        img=img.resize((144,144)) # new width & height
        img=ImageTk.PhotoImage(img)
        e1 =tk.Label(root)
        e1.grid(row=row,column=col,pady=100,padx=10)
        e1.image = img
        text_answer=prediction_RF_input_Normal
        text_answer=text_answer.tolist()
        l2 = tk.Label(root,text=text_answer,width=20,font=my_font2)  
        l2.grid(row=row+1,column=col,pady=0,padx=10)
        e1['image']=img # garbage collection
        if(col==7): # start new line after third column
            row=row+2# start wtih next row
            col=1    # start with first column
        else:       # within the same row 
            col=col+1 # increase to next column                 
root.mainloop()  # Keep the window open
