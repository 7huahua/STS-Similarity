from Total_similarity import TotalSimilarity
from ST_similarity import STSimilarity
from Semc_similarity import SemcSimilarity
import pandas as pd
import numpy as np


def compare_resample_method(stayregion_path,profile_path,resample_data,resample_num,resample_weight,segement_method):
    
    # the file name is like:
    # [userdata]:alluser/dropped/sample
    # [sample number]:bymean/bymedian
    # [sample weight]:bylocalweight/bytfidf

    pre_name = 'data/evaluate/pictures/'+resample_data+'_'+str(resample_num)+'_'+str(resample_weight)+'_'+segement_method

    # get st_sim
    st_similarity = STSimilarity(stayregion_path,feature_types='location')
    st_sim = st_similarity.get_st_similarity('cosine')

    # get semc_sim 
    semc_similarity = SemcSimilarity(profile_path=profile_path)
    userprofile = semc_similarity.user_profile
    semc_sim = semc_similarity.compute_semc_similarity(userprofile)

    # get ground truth
    total_sim = TotalSimilarity()
    stayregions = pd.read_csv(stayregion_path)
    total_sim.get_ground_truth(stayregions)

    # compute total stsemc_sim
    total_sim.compute_total_similarity(st_sim=st_sim,semc_sim=semc_sim,w1=0.5,w2=0.5)
    y_true_stsemc, y_pred_stsemc = total_sim.dict_to_array(total_sim.ground_truth,total_sim.total_sim_dict)

    # compute only semc_sim
    total_sim.compute_total_similarity(st_sim=None,semc_sim=semc_sim,w1=0.5,w2=0.5)
    y_true_onlysemc, y_pred_onlysemc = total_sim.dict_to_array(total_sim.ground_truth,total_sim.total_sim_dict)

    # compute only st_sim
    total_sim.compute_total_similarity(st_sim=st_sim,semc_sim=None,w1=0.5,w2=0.5)
    y_true_onlyst, y_pred_onlyst = total_sim.dict_to_array(total_sim.ground_truth,total_sim.total_sim_dict)

    # evaluate
    total_sim = TotalSimilarity()

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkcyan', 'magenta', 'gold']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'H']
    linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    x_axis = range(1,20)

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
    profile_path = 'data/user_profile/selected2_mean_local_week.pkl'
    compare_resample_method(stayregion_path=stayregion_path,profile_path=profile_path,
                            resample_data='selected2',
                            resample_num='mean',
                            resample_weight='local',
                            segement_method='week_')
    