# STS-Similarity


This project is for my paper: [Mining User Similarity from GPS Trajectory Based on Spatial-temporal and Semantic Information] (https://ieeexplore.ieee.org/document/9874192/metrics#metrics)


ST_similarity 
This file includes location features (could include point features ellipse features in the future)
Location features compute the tfidf vector of the clusters of stay points.


Semc_similarity
Semc firstly generate the semantic sequences from stay region clusters, then use prefixspan compute the maximal frequent sequence patterns, finally compute the similarity of mfsp from every 2 users as their semantic similarity.

More details please refer to the paper.
If you wanna use this code, please refer this paper in your research properlly.