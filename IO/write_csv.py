import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from os.path import split
from sklearn.preprocessing import LabelEncoder


def write_csv(dataframe):
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

#Test Function
#dictt = {'lakkjkjkj.png':['a','a','a','a','a']}
#df = pd.DataFrame.from_dict(dictt)
#write_csv(df)
