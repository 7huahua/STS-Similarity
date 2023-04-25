import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

class LocationFeatures(object):
    def __init__(self,staypoints=None,label='db_label'):
        self.label = label
        self.staypoints = self.get_locations(staypoints)

        # location features are stored in CSR format
        self.users, self.location_features = self.get_location_features(self.staypoints)

    def get_locations(self,staypoints):
        coords = staypoints[['lat','lng']].values
        kms_per_radian = 6371.0086
        epsilon = 0.1 / kms_per_radian
        
        db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

        staypoints[self.label] = db.labels_

        return staypoints

    def get_location_features(self,staypoints):

        # using TF-IDF score as the location features of every user
        locations = staypoints[staypoints[self.label] != -1].groupby('user').apply(lambda x:' '.join(str(i) for i in x[self.label].values)).values
        vectorizer = TfidfVectorizer(stop_words = None,token_pattern='(?u)\\b\\w+\\b')
        sparse_matrix = vectorizer.fit_transform(locations)

        user_list = staypoints[staypoints[self.label] != -1].user.unique()

        # sparse_matrix is stored in CSR format
        if isinstance(sparse_matrix, csr_matrix):
            print('sparse_matrix is stored in CSR format')
        else:
            print('sparse_matrix is not stored in CSR format')
            sparse_matrix = csr_matrix(sparse_matrix)

        row_norms = sparse_matrix.power(2).sum(axis=1).A1  # Compute row norms of X_norm
        is_normalized = all(abs(row_norms - 1) < 1e-6)  # Check if row norms are equal to 1

        if is_normalized:
            print('X_norm is already normalized')
        else:
            print('X_norm is not normalized')
            sparse_matrix = normalize(sparse_matrix, norm='l2')

        print(sparse_matrix.shape)

        # return matrix and user list

        return sparse_matrix, user_list
    

