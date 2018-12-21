# final-project-movielens-recommendation

 The goal of a recommendation system is to help users finding desired items faster. Personalized recommendations for movies are offered as ranked lists of movies. In performing this ranking, our recommendation system is trying to predict what the most suitable movies or actors are, based on the user’s previous experiences and ratings. 

 Based on this objective, we are going to build a Movie Recommendation System using MovieLens datasets.  

### Datasets: MovieLens  

The full datasets can be found [here](https://grouplens.org/datasets/movielens/).

For this project, we used the full dataset of MovieLens which has 27,753,444 ratings and 1,128 genomes applied to 53,889 movies by 283,228 users. Last updated 9/2018.

We used movieId, userId and ratings to make the first predictions, and then add genomes as the side information.

### Algorithms:

- Surprise: K-NN
- Surprise: SVDpp
- Factorization Machine: using lightfm
- Variational autoencoders for collaborative filtering

### Major steps:

1. Data preparation and spliting train set and test set

2. Model Train

3. Model evaluation: 
   ​     *ranking accuracy: recall, normalized discounted cumulative gain (NDCG)

   ​     *catalog coverage

# Requirements: 

- Python 3.6
- Jupyter Notebook

### Toolkits:

- [pandas](https://pandas.pydata.org)
- [numpy](http://www.numpy.org)
- [matplotlib](https://matplotlib.org)
- [seaborn](https://seaborn.pydata.org)
- [surprise](http://surpriselib.com)
- [sklearn](http://scikit-learn.org/stable/)
- [lightfm](https://lyst.github.io/lightfm/docs/home.html)
- [tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)



