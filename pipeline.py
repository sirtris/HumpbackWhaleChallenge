from IO.import_data import import_training_data
from IO.cross_val import cross_val
import classifiers.SVM_Classifier as clf

import IO.write_csv as write


def main():
    dataframe = import_training_data()                          # import the data
    classifier = clf.RF(max_depth=2)                            # set classifier to the classifier of your choice
    features = clf.extract_features(dataframe['Image'])         # extract features if necessary
    labels = dataframe['Id'].tolist()                           # extract the labels from the dataframe
    scores = cross_val(classifier, features, labels, nfolds=4) # use cross validation if necessary
    print(scores)
    """
    trim_data = classifier.extract(dataframe)
    scores = cross_val_score(classifier,trim_data,10)
    write_csv.write_csv(dataframe['Id'])
    """

if __name__ == '__main__':
    main()

