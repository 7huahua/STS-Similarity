import pandas as pd
from get_poi import * 

def process_row(row, tags, types):
    data = row.replace(" ", "")
    buffer = ""
    processed_data = ""
    index = 0

    while index < len(data):
        char = data[index]
        buffer += char

        # 其实我们只需要判断三种情况：
        # 0. 下一个字符不存在，说明到头了，记录buffer并清空buffer
        if index == len(data) - 1:
            processed_data += buffer
            buffer = ""
        
        else:
            next_char = data[index + 1]
        
        # 1. 如果buffer在tags中，并且下一个字符不是分号，说明buffer是一个完整的tag，记录buffer并清空buffer
        # 即使是门址的情况，这样也是可以检测出的
        if buffer in tags and next_char != ';':
            # print(buffer)  # 用于debug
            processed_data += buffer + ','
            buffer = ""
        
        # 2. 如果buffer包括分号，并且用分号split的后半部分在types中，说明buffer是一个完整的tag;type的形式，记录buffer并清空buffer
        elif ';' in buffer and buffer.split(';')[1] in types:
            # 如果是包含“门”的情况需要另外考虑，因为“门址点”和“门”是一种type
            if '门' in buffer and next_char == "址":
                buffer+=next_char+"点"
                index+=2
            # print(buffer)  # 用于debug
            processed_data += buffer + ','
            buffer = ""

        index += 1

    return processed_data

def process_poi3_row(row,tags,types):
    # poi3的数据比较容易处理，只需要使用"/"把字符串分隔开
    data = row.replace(" ", "")
    processed_data = ""
    for item in data.split("/"):
        if item in tags or item in types:
            processed_data += item + ','
    return processed_data


if __name__ == "__main__":
    types = ['中餐厅', '外国餐厅', '小吃快餐店', '蛋糕甜品店', '咖啡厅', '茶座', '酒吧', '星级酒店', '快捷酒店', '公寓式酒店', '民宿', '购物中心', '百货商场', '超市', '便利店', '家居建材', '家电数码', '商铺', '市场', '通讯营业厅', '邮局', '物流公司', '售票处', '洗衣店', '图文快印店', '照相馆', '房产中介机构', '公用事业', '维修点', '家政服务', '殡葬服务', '彩票销售点', '宠物服务', '报刊亭', '公共厕所', '步骑行专用道驿站', '美容', '美发', '美甲', '美体', '公园', '动物园', '植物园', '游乐园', '博物馆', '水族馆', '海滨浴场', '文物古迹', '教堂', '风景区', '景点', '寺庙', '度假村', '农家院', '电影院', 'ktv', '剧院', '歌舞厅', '网吧', '游戏场所', '洗浴按摩', '休闲广场', '体育场馆', '极限运动场所', '健身中心', '高等院校', '中学', '小学', '幼儿园', '成人教育', '亲子教育', '特殊教育学校', '留学中介机构', '科研机构', '培训机构', '图书馆', '科技馆', '新闻出版', '广播电视', '艺术团体', '美术馆', '展览馆', '文化宫', '综合医院', '专科医院', '诊所', '药店', '体检机构', '疗养院', '急救中心', '疾控中心', '医疗器械', '医疗保健', '汽车销售', '汽车维修', '汽车美容', '汽车配件', '汽车租赁', '汽车检测场', '飞机场', '火车站', '地铁站', '地铁线路', '长途汽车站', '公交车站', '公交线路', '港口', '停车场', '加油加气站', '服务区', '收费站', '桥', '充电站', '路侧停车位', '普通停车位', '接送点', '银行', 'ATM', 'atm', '信用社', '投资理财', '典当行', '写字楼', '住宅区', '宿舍', '内部楼栋', '公司', '园区', '农林园艺', '厂矿', '中央机构', '各级政府', '行政单位', '公检法机构', '涉外机构', '党派团体', '福利机构', '政治教育机构', '社会团体', '民主党派', '居民委员会', '高速公路出口', '高速公路入口', '机场出口', '机场入口', '车站出口', '车站入口', '停车场出入口', '自行车高速出口', '自行车高速入口', '自行车高速出入口', '岛屿', '山峰', '水系', '省', '省级城市', '地级市', '区县', '商圈', '乡镇', '村庄', '门址点', '门','新冠疫苗接种点', '其他','路口','绿地']
    tags = ["美食", "酒店", "购物", "生活服务", "丽人", "旅游景点", "休闲娱乐", "运动健身", "教育培训", "医疗", "汽车服务", "交通设施", "金融", "房地产", "公司企业", "政府机构", "自然地物", "行政地标","文化传媒", "出入口", "道路","门址","未知","life"]


# # 接下来是处理poi1和poi2的部分，最后获得了9982行

#     # 请注意，这边有一个坑，旅游景点;风景区和旅游景点;风景区绿地是两个不同的类别，但是后者只出现了一次，所以直接把原文件中的删除了。
#     # 类似的还有“行政区划”，包括“公司企业；公司行政区划” “政府机构；行政单位行政区划”也删除了
#     # read poi data
#     df = pd.read_csv('./data/poi1_poi2.csv')
#     df['processed_data'] = df['poi'].apply(lambda x: process_row(x, tags, types))
    
#     # get category1 and category2
#     # for every poi in x splited by ",", if poi can be split by ';', then category1 should add the first element, otherwise add poi
#     df['category1'] = df['processed_data'].apply(lambda x: ','.join(i for i in [poi.split(';')[0] if ';' in poi else poi for poi in x.split(',')]))
#     df['category2'] = df['processed_data'].apply(lambda x: ','.join(i for i in [poi.split(';')[1] if ';' in poi else '' for poi in x.split(',')]))

#     # drop useless columns
#     df.drop(['poi', 'processed_data'], axis=1, inplace=True)

#     df.to_csv('./data/poi_processed.csv', index=False)
#     df = pd.read_csv('./data/poi_processed.csv')
#     res = {}
#     cat = df.category1.values
#     for str in cat:
#         l = str.split(',')
#         for s in l:
#             if s not in res:
#                 res[s] = 1
#             else:
#                 res[s]+=1
        
#     print(res)
#     print(len(res))
#     # print the sorted result, in reverse order
#     print(sorted(res.items(), key=lambda x: x[1], reverse=True))

#     res2 = {}
#     cat2 = df.category2.values
#     for str in cat2:
#         l = str.split(',')
#         for s in l:
#             if s not in res2:
#                 res2[s] = 1

#             else:
#                 res2[s]+=1
        
#     print(res2)
#     print(len(res2))
#     print(sorted(res2.items(), key=lambda x: x[1], reverse=True))

# # 处理poi3的部分，最后获得了10193行
#     # 读取poi3的数据
#     df = pd.read_csv('./data/poi3.csv')
#     # 首先需要把poi3列的数据处理一下，如果有空格则去掉
#     df['poi3'] = df['poi3'].apply(lambda x: x.replace(' ', ''))
#     # 如果poi3列的数据中有“未知”，则把所有的未知后面加上逗号
#     df['poi3'] = df['poi3'].apply(lambda x: x.replace('未知', '未知,'))
#     # poi3的数据可以直接处理，使用“/”分割后再使用“;”分割，前半部分为category1，后半部分为category2
#     # for every poi in x splited by ",", if poi can be split by ';', then category1 should add the first element, otherwise add poi
#     df['category1'] = df['poi3'].apply(lambda x: ','.join(i for i in [poi.split(';')[0] if ';' in poi else poi for poi in x.split('/')]))
#     df['category2'] = df['poi3'].apply(lambda x: ','.join(i for i in [poi.split(';')[1] if ';' in poi else '' for poi in x.split('/')]))

    
#     df.drop(['poi3'], axis=1, inplace=True)
#     df.to_csv('./data/poi3_processed.csv', index=False)
#     print(df.shape)

    # 接下来，我们将两个文件合并，得到最终的poi文件
    df1 = pd.read_csv('./data/poi_processed.csv')
    df2 = pd.read_csv('./data/poi3_processed.csv')

    # 读取最初的staypoint文件，用来存储最后的结果
    df = pd.read_csv('./data/stay_regions.csv')
    print(df.shape)
   
    # 遍历df的每一行，我们需要对比其和另外两个文件的user,lat_min,lat_max,lng_min,lng_max,arr_t,lea_t,lat,lng,radius
    # 如果df1和df2都存在，那么把df1的category1加上df2的category1，df1的category2加上df2的category2，存储到df中
    # 如果df1或者df2存在，那么把存在的那个加到df中
    # 如果都不存在，那么就不加
    # 将df1和df2设置为多级索引，以便更轻松地查找和比较数据
    df1 = df1.set_index(['user', 'lat_min', 'lat_max', 'lng_min', 'lng_max', 'arr_t', 'lea_t', 'lat', 'lng', 'radius'])
    df2 = df2.set_index(['user', 'lat_min', 'lat_max', 'lng_min', 'lng_max', 'arr_t', 'lea_t', 'lat', 'lng', 'radius'])

    # 初始化df的新列
    df['category1'] = ''
    df['category2'] = ''

    # 遍历df的每一行，按照上述规则合并df1和df2的数据
    for idx, row in df.iterrows():
        index_values = (row['user'], row['lat_min'], row['lat_max'], row['lng_min'], row['lng_max'], row['arr_t'], row['lea_t'], row['lat'], row['lng'], row['radius'])

        # 检查df1和df2是否都包含所需数据
        in_df1 = index_values in df1.index
        in_df2 = index_values in df2.index



        # 根据数据的存在情况更新df的值
        if in_df1 and in_df2:
            # 输出要相加的两部分的type
            print(type(df1.loc[index_values, 'category1']))
            print(type(df1.loc[index_values, 'category2']))
            print(type(df2.loc[index_values, 'category1']))
            print(type(df2.loc[index_values, 'category2']))
            print(index_values)

            # 如果这四个值中有一个是nan，那么就把它变成空字符串
            if pd.isnull(df1.loc[index_values, 'category1']):
                df1.loc[index_values, 'category1'] = ''
            if pd.isnull(df1.loc[index_values, 'category2']):
                df1.loc[index_values, 'category2'] = ''
            if pd.isnull(df2.loc[index_values, 'category1']):
                df2.loc[index_values, 'category1'] = ''
            if pd.isnull(df2.loc[index_values, 'category2']):
                df2.loc[index_values, 'category2'] = ''
            
            df.loc[idx, 'category1'] = df1.loc[index_values, 'category1'] +','+ df2.loc[index_values, 'category1']
            df.loc[idx, 'category2'] = df1.loc[index_values, 'category2'] +','+ df2.loc[index_values, 'category2']
        elif in_df1:
            # print(type(df1.loc[index_values, 'category1']))
            # print(type(df1.loc[index_values, 'category2']))
            df.loc[idx, 'category1'] = df1.loc[index_values, 'category1']
            df.loc[idx, 'category2'] = df1.loc[index_values, 'category2']
        elif in_df2:
            # print(type(df2.loc[index_values, 'category1']))
            # print(type(df2.loc[index_values, 'category2']))
            df.loc[idx, 'category1'] = df2.loc[index_values, 'category1']
            df.loc[idx, 'category2'] = df2.loc[index_values, 'category2']

    # 保存结果到新的CSV文件
    df.to_csv('./data/combined_poi.csv', index=False)

