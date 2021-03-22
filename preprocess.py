import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

#Preprocessing the data
def preprocess():
    data = 'movies.csv'
    dataset = pd.read_csv(data)
    dataset.genres = dataset.genres.str.split('|')
    mlb = MultiLabelBinarizer()
    newdataset = mlb.fit_transform(dataset.genres)
    afterExpandingGenre = pd.DataFrame(data=newdataset, columns=mlb.classes_)
    combinedBeforeRating = pd.concat([dataset,afterExpandingGenre],axis = 1,sort = False )
    combinedAfterRating = pd.read_csv('ratings.csv')
    combinedBeforeRating = pd.merge(combinedBeforeRating,combinedAfterRating,on='movieId')
    combinedBeforeRating = combinedBeforeRating.groupby('movieId').mean().reset_index()
    combinedBeforeRating = pd.merge(dataset,combinedBeforeRating,on='movieId')
    combinedBeforeRating = combinedBeforeRating.drop(columns = ['genres','userId','timestamp'])
    combinedBeforeRating = pd.merge(combinedBeforeRating,pd.read_csv('budget.csv'),on = 'movieId')
    return combinedBeforeRating


