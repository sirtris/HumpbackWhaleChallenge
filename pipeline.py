from import_data import import_training_data
from sklearn.model_selection import cross_val_score
import classifier
import write_csv


def main():
    dataframe = import_training_data()
    trim_data = classifier.extract(dataframe)
    scores = cross_val_score(classifier,trim_data,10)
    write_csv.write_csv(dataframe['Id'])


if __name__ == '__main__':
    main()

