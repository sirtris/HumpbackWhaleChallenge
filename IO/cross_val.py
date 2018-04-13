from sklearn.model_selection import cross_val_score


def cross_val(clf, data, labels, nfolds = 10):
    # IMPORTANT: clf should have a .fit and a .predict function! data should be a dictionary with ['data'] as raw data
    # entry (for example a matrix with a lot of numbers).
    error_msg = "IMPORTANT: clf should have a .fit and a .predict function! data should be a matrix that represents the data (for example a feature matrix), and labels a vector with the corressponding labels."
    try:
        return cross_val_score(clf, data, labels, cv=nfolds)
    except:
        print(error_msg)



if __name__ == '__main__':
    # example code how to use the cross_val function:
    import random as r
    data = {}
    data['data'] = []
    data['labels'] = []
    for i in range(100):
        a = r.randint(0,1)
        b = r.randint(0,1)
        data['data'].append([a,b])
        if a + b == 2:
            data['labels'].append(1)
        else:
            data['labels'].append(0)
    # print(data['data'])

    # for a SVM:
    from sklearn import svm
    clf = svm.SVC()

    # for RF:
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=15, random_state=0)

    print(cross_val(clf, data['data'], data['labels'], nfolds=10))


