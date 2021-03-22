"""
1.Importing the data first
1. First one is data preprocessing and cealning the data
2.Combing the data and only taking the required field
3.Now we have to train our model using diffrent ml algorithms present in skleatn module
4.We will get the classification accuracy
5.Now will classify our own data
6.the final predicted value will be shown

"""
#importing libraries and necessary machine learning algorithm
import preprocess as pre
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scikitplot as skplt




if __name__ == '__main__':
    #seperating training data and test data
    final_data = pre.preprocess()
    features = final_data.drop(columns=[ 'rating', 'movieId', 'title' ]).values
    lables = final_data.rating.values

    lables2 = list(lables)
    for i in range(len(lables)):
        lables2[ i ] = lables[ i ].__round__(1)
        lables[ i ] = 0 if lables2[ i ] < 2.5 else 1
        lables2[ i ] = int(lables2[ i ].__round__(0))

    # Radial basis function
    # gamma Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    # n_estimators : no. of forest in the classification
    # Splitting the data
    features_train, features_test, lables_train, lables_test = train_test_split(features, lables, test_size=0.8)

    #List of machine learning algorithmn used
    lst = [ KNeighborsClassifier(), SVC(kernel="rbf", gamma='auto'), RandomForestClassifier(n_estimators=100),
            DecisionTreeClassifier(), MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=10000), GaussianNB() ]
    """no genres listed),Action,Adventure,Animation,Children,Comedy,Crime,Documentary,Drama,Fantasy,Film-Noir,Horror,IMAX,Musical,Mystery,Romance,Sci-Fi,Thriller,War,Western"""
    #my test data
    my_data = np.array([ [ 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69 ] ])
    print('Input data', my_data[ 0 ], sep=' : ')
    print('For predicting success or failure of a movie:')
    for i in range(len(lst)):
        #Training and predicting the success
        my_classifier = lst[ i ]
        my_classifier.fit(features_train, lables_train)
        prediction = my_classifier.predict(features_test)
        print(end='       ')
        print(type(my_classifier).__name__, end=" : ")
        print((my_classifier.score(features_test, lables_test) * 100).__round__(2))
        ans = 'Hit' if my_classifier.predict(my_data)[ 0 ] == 1 else 'Flop'
        print('         Predicted success: ', ans, sep=' ')
        trueval = list()
        predictval = list()
        for i in range(len(lables_test)):
            if lables_test[i] == 1:
                trueval.append('Hit')
            else:
                trueval.append('Flop')
            if prediction[i] == 1:
                predictval.append('Hit')
            else:
                predictval.append('Flop')
        #Plotting confusion matrix
        skplt.metrics.plot_confusion_matrix(trueval, predictval, normalize=True)
        plt.title('Confusion Matrix for '+type(my_classifier).__name__)
        plt.xlabel('Predicted Success')
        plt.ylabel('True Success')
        plt.show()
    print()
    lables = final_data.rating.values
    for i in range(len(lables)):
        lables[ i ] = int(lables[ i ].__round__(2))
    features_train, features_test, lables_train, lables_test = train_test_split(features, lables2, test_size=0.8)
    print('For exact rating:')
    for i in range(len(lst)):
        # Training and predicting the outcome
        my_classifier = lst[ i ]
        my_classifier.fit(features_train, lables_train)
        prediction = my_classifier.predict(features_test)
        print(end='       ')
        print(type(my_classifier).__name__, end=" : ")
        print((my_classifier.score(features_test, lables_test) * 100).__round__(2))
        print('         Predicted rating: ', my_classifier.predict(my_data)[ 0 ], sep=' ')

        # Plotting confusion matrix
        skplt.metrics.plot_confusion_matrix(lables_test, prediction, normalize=True)
        plt.title('Confusion Matrix for ' + type(my_classifier).__name__)
        plt.xlabel('Predicted Rating')
        plt.ylabel('True Rating')
        plt.show()
