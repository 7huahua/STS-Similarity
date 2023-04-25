from Total_similarity import TotalSimilarity
from ST_similarity import STSimilarity
from Semc_similarity import SemcSimilarity
import pandas as pd
import numpy as np


def compare_resample_method(stayregion_path,profile_path,resample_data,resample_num,resample_weight,segement_method):
    
    # 为了区别每一次的结果，文件名需要统一命名，以下为需要考虑的几种情况
    # folder1:alluser/droppeduser
    # folder2:bymean/bymedian/by100
    # folder3:bylocalweight/byglobalweight/bytfidf

    pre_name = 'data/evaluate/pictures/'+resample_data+'_'+str(resample_num)+'_'+str(resample_weight)+'_'+segement_method
    # 接下来的所有文件名按照上述顺序命名

    # 获取st_sim
    st_similarity = STSimilarity(stayregion_path,feature_types='location')
    st_sim = st_similarity.get_st_similarity('cosine')


    # 获取semc_sim 
    semc_similarity = SemcSimilarity(profile_path=profile_path)
    userprofile = semc_similarity.user_profile
    # 从dictionary中删除key为29和229的user
    semc_sim = semc_similarity.compute_semc_similarity(userprofile)

    # 获取ground truth
    total_sim = TotalSimilarity()

    stayregions = pd.read_csv('data/stay_regions.csv')
    # drop user 29 and user 229
    stayregions = stayregions[stayregions.user != 29]
    stayregions = stayregions[stayregions.user != 229]
    total_sim.get_ground_truth(stayregions)

    # 计算stsemc_sim
    total_sim.compute_total_similarity(st_sim=st_sim,semc_sim=semc_sim,w1=0.5,w2=0.5)
    y_true_stsemc, y_pred_stsemc = total_sim.dict_to_array(total_sim.ground_truth,total_sim.total_sim_dict)

    # 计算semc_sim
    total_sim.compute_total_similarity(st_sim=None,semc_sim=semc_sim,w1=0.5,w2=0.5)
    y_true_onlysemc, y_pred_onlysemc = total_sim.dict_to_array(total_sim.ground_truth,total_sim.total_sim_dict)

    # 计算st_sim
    total_sim.compute_total_similarity(st_sim=st_sim,semc_sim=None,w1=0.5,w2=0.5)
    y_true_onlyst, y_pred_onlyst = total_sim.dict_to_array(total_sim.ground_truth,total_sim.total_sim_dict)

    # 评估
    total_sim = TotalSimilarity()

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkcyan', 'magenta', 'gold']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'H']
    linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    x_axis = range(1,200)

    maps_onlysemc = []
    ndcgs_onlysemc = []
    for i in x_axis:
        MAP_k, NDCG_k = total_sim.evaluate(y_true_onlysemc,y_pred_onlysemc,i)
        maps_onlysemc.append(MAP_k)
        ndcgs_onlysemc.append(NDCG_k)

    maps_onlyst = []
    ndcgs_onlyst = []
    for i in x_axis:
        MAP_k, NDCG_k = total_sim.evaluate(y_true_onlyst,y_pred_onlyst,i)
        maps_onlyst.append(MAP_k)
        ndcgs_onlyst.append(NDCG_k)

    maps_stsemc = []
    ndcgs_stsemc = []
    for i in x_axis:
        MAP_k, NDCG_k = total_sim.evaluate(y_true_stsemc,y_pred_stsemc,i)
        maps_stsemc.append(MAP_k)
        ndcgs_stsemc.append(NDCG_k)

    # plot
    maps = [maps_onlysemc,maps_onlyst,maps_stsemc]
    ndcgs = [ndcgs_onlysemc,ndcgs_onlyst,ndcgs_stsemc]
    labels = ['only semc','only st','st+semc']

    total_sim.plot_multiple_lines(x_axis,maps,labels,colors,markers,linestyles,filename=pre_name+'map_200.png')
    total_sim.plot_multiple_lines(x_axis,ndcgs,labels,colors,markers,linestyles,filename=pre_name+'ndcg_200.png')

def compare_week_weekday():
    pass


if __name__ == '__main__':
    stayregion_path = 'data/stay_regions.csv'
    profile_path = 'data/user_profile/dropped_mean_local_week_sup2.pkl'
    compare_resample_method(stayregion_path=stayregion_path,profile_path=profile_path,
                            resample_data='dropped',
                            resample_num='mean',
                            resample_weight='local',
                            segement_method='week_sup2')
    # compare_week_weekday()

    # 