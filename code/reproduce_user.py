# 这个文件用于重新处理用户数据，包括删选4天2周的用户、去除pattern过少的用户、对于pattern过多的用户下采样、过少的上采样
# 首先，导入需要的库
import pandas as pd
import numpy as np
import os


# 检查用户pattern数量是否符合要求
def check_pattern_num(df, min_pattern_num, max_pattern_num):
    pass

# 筛选用户
def filter_user(src, min_pattern_num, max_pattern_num, dst):
    pass

# 下采样
def down_sample(src, dst):
    pass

# 上采样
def up_sample(src, dst):
    pass

if __name__ == "__main__":
    src = 'data/combined_poi.csv'
    