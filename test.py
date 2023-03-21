from haversine import haversine, Unit
import pandas as pd

# 定义函数计算中心到顶点的距离
def calc_radius(lat_min, lat_max, lng_min, lng_max, center_lat, center_lng):
    # 计算顶点坐标
    vertex1 = (lat_min, lng_min)
    vertex2 = (lat_max, lng_max)
    # 计算中心点坐标
    center = (center_lat, center_lng)
    # 计算中心点到两个顶点的距离
    dist1 = haversine(center, vertex1, unit=Unit.METERS)
    dist2 = haversine(center, vertex2, unit=Unit.METERS)
    # 取最大值作为半径
    radius = max(dist1, dist2)
    return radius

# 示例调用
lat_min = 39.905
lat_max = 39.925
lng_min = 116.365
lng_max = 116.385
center_lat = 39.915
center_lng = 116.375

radius = calc_radius(lat_min, lat_max, lng_min, lng_max, center_lat, center_lng)
print(radius)


df = pd.read_csv('data/stay_regions.csv')
# df['radius'] = df.apply(lambda x: calc_radius(x['lat_min'], x['lat_max'], x['lng_min'], x['lng_max'], x['lat'], x['lng']), axis=1)
# df.to_csv('data/stay_regions.csv', index=False)
print(df.radius.max())