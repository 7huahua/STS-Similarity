import pandas as pd
import numpy as np
# import skmob
from sklearn.cluster import DBSCAN
class PointFeatures:
    def __init__(self, staypoints):
        pass
        self.staypoints = self.get_locations(staypoints)

        self.point_features = self.get_point_features()


    def get_staypoint_number(self, staypoints):
        return staypoints.groupby('user').apply(lambda x: x.user.count())
    
    def get_staypoint_entropy(self, staypoints):
        pass
        return
    
    def get_staypoint_rog(self, staypoints):
        pass

    def get_locations(self,staypoints):
        coords = staypoints[['lat','lng']].values
        kms_per_radian = 6371.0086
        epsilon = 0.1 / kms_per_radian
        
        db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

        staypoints['db_labels'] = db.labels_

        return staypoints

    def get_home_location(self, staypoints):
        pass

    def get_work_locations(self, staypoints):
        pass

    def get_point_features(self):
        pass
        self.staypoint_number = self.get_staypoint_number(self.staypoints)
        self.staypoint_entropy = self.get_staypoint_entropy(self.staypoints)
        self.staypoint_rog = self.get_staypoint_rog(self.staypoints)
        self.home_location = self.get_home_location(self.staypoints)
        self.work_location = self.get_work_location(self.staypoints)

        self.point_features = pd.concat([self.staypoint_number, 
                                    self.staypoint_entropy,
                                    self.staypoint_rog,
                                    self.home_location,
                                    self.work_location],axis = 1)
        
        return self.point_features