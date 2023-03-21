import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from point_features import PointFeatures
from ellipse_features import EllipseFeatures
from location_features import LocationFeatures
from scipy.sparse import hstack
from sklearn.preprocessing import normalize

class STSimilarity(object):
    def __init__(self, src = None, feature_types = None):
        self.stayregions = self.load_stay_regions(src)
        self.feature_types = feature_types or ['point', 'ellipse', 'location']

        self.point_features = self.get_point_features(self.stayregions) if 'point' in self.feature_types else None
        self.ellipse_features = self.get_ellipse_features(self.stayregions) if 'ellipse' in self.feature_types else None
        self.location_features = self.get_location_features(self.stayregions) if 'location' in self.feature_types else None
        # self.st_similatiry = self.get_st_similarity(self.location_features)

    def load_stay_regions(self, src):
        df = pd.read_csv(src,parse_dates = ['arr_t','lea_t'])
        if not df.shape[0]:
            raise ValueError
        return df
    
    def get_point_features(self,stay_regions):
        pf_obj = PointFeatures(stay_regions)
        point_features = pf_obj.get_point_features(stay_regions)
        return point_features
    
    def get_ellipse_features(self, stay_regions):
        ef_obj = EllipseFeatures(stay_regions)
        ellipse_features = ef_obj.get_ellipse_features(stay_regions)
        return ellipse_features
    
    def get_location_features(self, stay_regions):
        lf_obj = LocationFeatures(stay_regions)
        location_features = lf_obj.get_location_features(stay_regions)
        return location_features
    
    def get_st_similarity(self, metric = 'cosine'):
        if self.point_features is None and self.ellipse_features is None and self.location_features is None:
            raise ValueError('At least one feature is required to compute similarity')

        # Concatenate selected features into a single matrix
        features = []
        if self.point_features is not None:
            features.append(self.point_features)
        if self.ellipse_features is not None:
            features.append(self.ellipse_features)
        if self.location_features is not None:
            features.append(self.location_features)

        X = hstack(features, format='csr')
        
        # Normalize X
        X_norm = normalize(X, norm='l2', axis=1)
        
        # Compute similarity matrix
        if metric == 'cosine':
            sim_matrix = 1/(1+cosine_distances(X_norm))
        elif metric == 'euclidean':
            sim_matrix = 1/(1+euclidean_distances(X_norm))
        else:
            raise ValueError('Invalid similarity metric: {}'.format(metric))
        
        return sim_matrix
    

if __name__ == '__main__':
    st_obj = STSimilarity('data/stay_regions.csv','location')
    st_similarity = st_obj.get_st_similarity('cosine')
    print(st_similarity)
    print(np.mat(st_similarity).shape)