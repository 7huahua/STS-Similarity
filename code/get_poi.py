import pandas as pd
import numpy as np
import requests
import json
import os

class GetPoi(object):
    def __init__(self,src = None):
        self.staypoints = self.load_staypoints(src)
        self.url = "https://api.map.baidu.com/place/v2/search"

    def load_staypoints(self,src):
        # load stay points
        df = pd.read_csv(src,parse_dates = ['arr_t','lea_t'])
        if not df.shape[0]:
            raise ValueError
        return df[3655:]
     
    def load_params(self,location,radius,query):
        params = {
            'query': query,
            'location': location,
            'radius': radius,
            'scope': '2',
            'output': 'json',
            'ak': 'X2pBOlsVEZwjjIKXHP0oXKU35wSz1fLs'
            'page'
        }
        return params
        

    def _get_poi(self,location,radius,query):
        #get poi for every stay point

        page_size = 20
        total = 0
        page_num = 0
        result = []

        while True:
            # 构造请求参数
            params = {
                "query": query,
                "location": location,
                "radius": radius,
                'scope': '2',
                'output': 'json',
                'ak': 'X2pBOlsVEZwjjIKXHP0oXKU35wSz1fLs',
                "page_size": len(query)*page_size,
                "page_num": page_num
            }
            # 发送请求
            response = requests.get(self.url, params=params)
            # 解析json数据
            data = json.loads(response.text)
            # 处理返回结果
            status = data["status"]
            message = data["message"]
            if status == 0 and message == "ok":
                results = data["results"]
                for poi in results:
                    tag = poi['detail_info'].get('tag')
                    poi_type = poi['detail_info'].get('type')
                    result += ',' + tag if tag else poi_type if poi_type else '未知'
                
                total += len(results)
                # 如果本次请求的结果数小于page_size，也就是说一个达到20条的poi的query都没了
                # 则说明已经获取到所有记录，退出循环
                if len(results) < page_size:
                    break
                page_num += 1
            else:
                # 请求失败，抛出异常
                raise Exception(f"API访问失败：location = {location}, status={status}, message={message}")

        # print(f"共获取到{total}条记录。")
        # print(result)
        return " ".join([poi for poi in result])

    def get_poi(self):
        # get poi for every stay point
        categories1 = ["美食", "酒店", "购物", "生活服务", "丽人", "旅游景点", "休闲娱乐", "运动健身", "教育培训"]
        categories2 = ["医疗", "汽车服务", "交通设施", "金融", "房地产", "公司企业", "政府机构", "自然地物", "行政地标"]
        categories3 = ["文化传媒", "出入口", "门址"]
        query1 = "$".join(categories1)
        query2 = "$".join(categories2)
        query3 = "$".join(categories3)

        for index, row in self.staypoints.iterrows():
            try:
                # poi1 = self._get_poi(location=str(row['lat']) + ',' + str(row['lng']), radius=str(row['radius']), query=query1)
                # self.staypoints.loc[index, 'poi1'] = poi1
                # poi2 = self._get_poi(location=str(row['lat']) + ',' + str(row['lng']), radius=str(row['radius']), query=query2)
                # self.staypoints.loc[index, 'poi2'] = poi2
                poi3 = self._get_poi(location=str(row['lat']) + ',' + str(row['lng']), radius=str(row['radius']), query=query3)
                self.staypoints.loc[index, 'poi3'] = poi3

            except:
                self.staypoints.to_csv('data/poi3.csv',index = False)
                print(f'index: {index}, row: {row}')
                raise Exception('API请求失败，终止函数')
            
        self.staypoints.to_csv('data/poi3.csv',index = False)
        print(f'index: {index}, row: {row}')
        print('运行完成')

    
    def get_poi_df(self):
        # 从./data中读取所有以poi开头的csv文件，并仅仅保留poi1和poi2两列有值的数据
        # 首先，读取所有的csv文件
        all_df = pd.DataFrame()
        for file in os.listdir('./data'):
            if file.startswith('poi_') and file!='poi_processed.csv':
                df = pd.read_csv('./data/' + file)
                # 然后，仅仅保留poi1或者poi2两列有值并且值不为nan的数据
                df = df[(df['poi1'].notnull()) | (df['poi2'].notnull())]
                all_df = pd.concat([all_df, df])
        
        # 最后，将poi1和poi2两列合并为一个列,以“ ”分隔
        all_df['poi'] = all_df['poi1'] + " " + all_df['poi2']
        # 删除poi1和poi2两列
        all_df.drop(['poi1', 'poi2'], axis=1, inplace=True)
        # 再次检查all_df中是否有空值或者nan值
        all_df = all_df[all_df['poi'].notnull()]
        return all_df

    

if __name__ == '__main__':
    src = 'data/stay_regions.csv'
    poi_obj = GetPoi(src)
    # staypoints = poi_obj.staypoints
    # get poi for every stay point
    poi_obj.get_poi()
    # print(staypoints.shape)
    # df = poi_obj.get_poi_df()
    # print(df.shape)
    # # 保存到csv文件
    # df.to_csv('data/poi1_poi2.csv',index = False)
    