#自作モジュール
import modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


"""
読み込み→標準化→(グラフ描画)→安定度、深さを出す

【問題点】
webcam_demo.pyでsquat2の読み込みが途中で終わってしまう。
動画の秒数が取得できれば負荷率が出せたり幅が広がる→ffmpegというモジュールを使えばなんとかなる？

"""

if __name__ == "__main__":
    #読み込み
    squat1_x_df = pd.read_csv(f'../images/squat1_x_df.csv',index_col=0)
    squat1_y_df = pd.read_csv(f'../images/squat1_y_df.csv',index_col=0)
    squat2_x_df = pd.read_csv(f'../images/squat2_x_df.csv',index_col=0)
    squat2_y_df = pd.read_csv(f'../images/squat2_y_df.csv',index_col=0)


    #標準化
    squat1_x_df = modules.replace_second(modules.standarizatioin_df(squat1_x_df))
    squat1_y_df = modules.replace_second(modules.standarizatioin_df(squat1_y_df))
    squat2_x_df = modules.replace_second(modules.standarizatioin_df(squat2_x_df))
    squat2_y_df = modules.replace_second(modules.standarizatioin_df(squat2_y_df))


    #該当のdfをグラフ化
    modules.draw_graph(squat1_y_df)
    modules.draw_graph(squat2_y_df)


    #安定度、深さを出す関数
    modules.horizontal_stability(squat1_x_df)
    modules.horizontal_stability(squat2_x_df)
    modules.down_depth(squat1_y_df)
