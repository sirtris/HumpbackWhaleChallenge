import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from os.path import split
from sklearn.preprocessing import LabelEncoder


def write_csv_old(dataframe):
    #Takes a dataframe with a dictionary as data, with the image ID as the key
    #And the string values of the whale
    with open("submission.csv", "w") as f:
        f.write("Image,Id\n")
        imagedict = dataframe.to_dict('list')
        for key in imagedict:
            image = key
            predicted_tags = imagedict[key]
            stringtags = " ".join(predicted_tags)
            f.write("%s,%s\n" % (image, stringtags))

def write_csv(labels,classifications,force_new_whale=False):
    final_prediction = []
    for i in range(len(labels)):
        whales = []
        #Add the labels for every classifier
        for c in classifications:
            if(type(c[i]) == list):
                whales.extend(c[i])
            else:
                print(c[i])
                whales.append(c[i])
        #If new the flag is set to true and new whale is not in the list, add it at the start
        if(force_new_whale and "new_whale" not in whales):
            whales.insert(0,'new_whale')
        #Limit the classification length to 5
        if(len(whales) > 5):
            whales = whales[0:5]
        elif(len(whales) < 5):
            whales.append('new_whale')
        final_prediction.append(whales)
    #Write the whale IDs in the csv file
    with open("submissiontest.csv", "w") as f:
        f.write("Image,Id\n")
        for i in range(len(labels)):
            image = labels[i]
            predicted_tags = final_prediction[i]
            print(predicted_tags)
            stringtags = " ".join(predicted_tags)
            f.write("%s,%s\n" % (image, stringtags))

#Test Function
#dictt = {'lakkjkjkj.png':['a','a','a','a','a']}
#df = pd.DataFrame.from_dict(dictt)
#write_csv_old(df)

#Test new function
#labels = ['whale1','whale2','whale3']
#classifications = [['b','c','d'],[['b','c','d'],['b','c','d'],['b','c','d']]]
#write_csv(labels,classifications,force_new_whale=False)
