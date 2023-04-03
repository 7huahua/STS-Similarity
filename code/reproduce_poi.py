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


if __name__ == "__main__":
    types = ['中餐厅', '外国餐厅', '小吃快餐店', '蛋糕甜品店', '咖啡厅', '茶座', '酒吧', '星级酒店', '快捷酒店', '公寓式酒店', '民宿', '购物中心', '百货商场', '超市', '便利店', '家居建材', '家电数码', '商铺', '市场', '通讯营业厅', '邮局', '物流公司', '售票处', '洗衣店', '图文快印店', '照相馆', '房产中介机构', '公用事业', '维修点', '家政服务', '殡葬服务', '彩票销售点', '宠物服务', '报刊亭', '公共厕所', '步骑行专用道驿站', '美容', '美发', '美甲', '美体', '公园', '动物园', '植物园', '游乐园', '博物馆', '水族馆', '海滨浴场', '文物古迹', '教堂', '风景区', '景点', '寺庙', '度假村', '农家院', '电影院', 'ktv', '剧院', '歌舞厅', '网吧', '游戏场所', '洗浴按摩', '休闲广场', '体育场馆', '极限运动场所', '健身中心', '高等院校', '中学', '小学', '幼儿园', '成人教育', '亲子教育', '特殊教育学校', '留学中介机构', '科研机构', '培训机构', '图书馆', '科技馆', '新闻出版', '广播电视', '艺术团体', '美术馆', '展览馆', '文化宫', '综合医院', '专科医院', '诊所', '药店', '体检机构', '疗养院', '急救中心', '疾控中心', '医疗器械', '医疗保健', '汽车销售', '汽车维修', '汽车美容', '汽车配件', '汽车租赁', '汽车检测场', '飞机场', '火车站', '地铁站', '地铁线路', '长途汽车站', '公交车站', '公交线路', '港口', '停车场', '加油加气站', '服务区', '收费站', '桥', '充电站', '路侧停车位', '普通停车位', '接送点', '银行', 'ATM', 'atm', '信用社', '投资理财', '典当行', '写字楼', '住宅区', '宿舍', '内部楼栋', '公司', '园区', '农林园艺', '厂矿', '中央机构', '各级政府', '行政单位', '公检法机构', '涉外机构', '党派团体', '福利机构', '政治教育机构', '社会团体', '民主党派', '居民委员会', '高速公路出口', '高速公路入口', '机场出口', '机场入口', '车站出口', '车站入口', '停车场出入口', '自行车高速出口', '自行车高速入口', '自行车高速出入口', '岛屿', '山峰', '水系', '省', '省级城市', '地级市', '区县', '商圈', '乡镇', '村庄', '门址点', '门','新冠疫苗接种点', '其他','路口','绿地']
    tags = ["美食", "酒店", "购物", "生活服务", "丽人", "旅游景点", "休闲娱乐", "运动健身", "教育培训", "医疗", "汽车服务", "交通设施", "金融", "房地产", "公司企业", "政府机构", "自然地物", "行政地标","文化传媒", "出入口", "道路","门址","未知","life"]

    # 请注意，这边有一个坑，旅游景点;风景区和旅游景点;风景区绿地是两个不同的类别，但是后者只出现了一次，所以直接把原文件中的删除了。
    # 类似的还有“行政区划”，包括“公司企业；公司行政区划” “政府机构；行政单位行政区划”也删除了
    # read poi data
    df = pd.read_csv('./data/poi1_poi2.csv')
    df['processed_data'] = df['poi'].apply(lambda x: process_row(x, tags, types))
    
    # get category1 and category2
    # for every poi in x splited by ",", if poi can be split by ';', then category1 should add the first element, otherwise add poi
    df['category1'] = df['processed_data'].apply(lambda x: ','.join(i for i in [poi.split(';')[0] if ';' in poi else poi for poi in x.split(',')]))
    df['category2'] = df['processed_data'].apply(lambda x: ','.join(i for i in [poi.split(';')[1] if ';' in poi else '' for poi in x.split(',')]))

    # drop useless columns
    df.drop(['poi', 'processed_data'], axis=1, inplace=True)

    df.to_csv('./data/poi_processed.csv', index=False)
    df = pd.read_csv('./data/poi_processed.csv')
    res = {}
    cat = df.category1.values
    for str in cat:
        l = str.split(',')
        for s in l:
            if s not in res:
                res[s] = 1
            else:
                res[s]+=1
        
    print(res)
    print(len(res))
    # print the sorted result, in reverse order
    print(sorted(res.items(), key=lambda x: x[1], reverse=True))

    res2 = {}
    cat2 = df.category2.values
    for str in cat2:
        l = str.split(',')
        for s in l:
            if s not in res2:
                res2[s] = 1

            else:
                res2[s]+=1
        
    print(res2)
    print(len(res2))
    print(sorted(res2.items(), key=lambda x: x[1], reverse=True))