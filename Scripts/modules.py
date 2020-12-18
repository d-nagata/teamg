import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing


def replace_second(df):
    """
    かかった秒数を列に追加

    Args:
        df (pd.DataFrame): データフレーム
    """
    print('df.index',df.index)
    print('df.shape',df.shape)
    df['second'] = df.index/df['length'].values[0]
    df = df.set_index('second')
    return df


def standarizatioin_df(df):
    """
    身長を1として標準化する
    身長は鼻とくるぶしまでの距離が一番大きかったものとする
    横幅に関しても同様

    Args:
        df (pd.DataFrame): データフレーム

    Returns:
        (pd.DataFrame): データフレーム
    """
    #身長: 鼻とくるぶしまでの距離が一番大きかったものを身長とする
    height_series_right = df.iloc[:,16]-df.iloc[:,0]
    height_series_left = df.iloc[:,15]-df.iloc[:,0]
    height_right = height_series_right.max()
    height_left = height_series_left.max()
    height = (height_right+height_left)/2


    for rep in range(17):
        df.iloc[:,rep] = (df.iloc[:,16] - df.iloc[:,rep]) / height
    return df


def draw_graph(df):
    """
    全17点のグラフ出力
    """
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    
    for i in range(df.shape[1]):
        ax.plot(df.index,df.iloc[:,i])
    
    ax.legend(['nose','leftEye','rightEye','leftEar','rightEar','leftShoulder',
                'rightShoulder','leftElbow','rightElbow','leftWrist','rightWrist',
                'leftHip','rightHip','leftKnee','rightKnee','leftAnkle','rightAnkle'], loc='lower left', borderaxespad=0, fontsize=18)

    plt.show()


def more_plotpoints(df1,df2):
    """
    'leftHip','rightHip'に限定。
    プロット点が足りないため、最小公倍数にプロット点を水増し

    Returns:
        pd.DataFrame : 水増し後のデータフレーム
    """
    
    #df1について、プロット点を線分の内分点で割増
    df1_l_list = []
    df1_r_list = []
    df1_ttf_l_list = []
    df1_ttf_r_list = []
    
    #rightとleftについてそれぞれ計算
    print(df1.shape,df2.shape)
    for i in range(df1.shape[0]-1):
        for j in range(df2.shape[0]):
            df1_l_list.append(((df2.shape[0]-j)*df1.iloc[i,0]+j*df1.iloc[i+1,0])/df1.shape[0])
            df1_r_list.append(((df2.shape[0]-j)*df1.iloc[i,1]+j*df1.iloc[i+1,1])/df1.shape[0])
            '''df1_ttf_l_list.append(((df2.shape[0]-j)*df1.iloc[i,2]+j*df1.iloc[i+1,2])/df1.shape[0])
            df1_ttf_r_list.append(((df2.shape[0]-j)*df1.iloc[i,3]+j*df1.iloc[i+1,3])/df1.shape[0])'''
    
    #df2について、プロット点を線分の内分点で割増
    df2_l_list = []
    df2_r_list = []
    df2_ttf_l_list = []
    df2_ttf_r_list = []
    
    #rightとleftについてそれぞれ計算
    for i in range(df2.shape[0]-1):
        for j in range(df1.shape[0]):
            df2_l_list.append(((df1.shape[0]-j)*df2.iloc[i,0]+j*df2.iloc[i+1,0])/df2.shape[0])
            df2_r_list.append(((df1.shape[0]-j)*df2.iloc[i,1]+j*df2.iloc[i+1,1])/df2.shape[0])
            '''df2_ttf_l_list.append(((df1.shape[0]-j)*df2.iloc[i,2]+j*df2.iloc[i+1,2])/df2.shape[0])
            df2_ttf_r_list.append(((df1.shape[0]-j)*df2.iloc[i,3]+j*df2.iloc[i+1,3])/df2.shape[0])'''
    
    #df作成
    '''
    df1 = pd.DataFrame([df1_l_list,df1_r_list,df1_ttf_l_list,df1_ttf_r_list],index=['leftHip','rightHip','fft_leftHip','fft_rightHip']).T
    df2 = pd.DataFrame([df2_l_list,df2_r_list,df2_ttf_l_list,df2_ttf_r_list],index=['leftHip','rightHip','fft_leftHip','fft_rightHip']).T
    '''
    df1 = pd.DataFrame([df1_l_list,df1_r_list],index=['leftHip','rightHip']).T
    df2 = pd.DataFrame([df2_l_list,df2_r_list],index=['leftHip','rightHip']).T
    print("df1:",df1)
    print("df2:",df2)
    
    #描画
    fig = plt.figure(figsize=(24,9))
    ax = fig.add_subplot(1,2,1)
    ax.plot(df1['leftHip'])
    ax.plot(df1['rightHip'])
    ax = fig.add_subplot(1,2,2)
    ax.plot(df2['leftHip'])
    ax.plot(df2['rightHip'])
    plt.show()

    return df1,df2




def crop_filter(df):
    """
    実際にスクワットをしている時間のみを切り出し

    Returns:
        pd.DataFrame : 切り出し後のデータフレーム
    """
    #初期レベルは一番座標が小さい所(画像では最高点)
    lowest_level_1 = int((df['leftHip'].idxmin() + df['rightHip'].idxmin()) / 2)
    highest_level =  int((df['leftHip'].idxmax() + df['rightHip'].idxmax()) / 2)
    lowest_level_2 = int((df['leftHip'][df['leftHip'].idxmax():].idxmin() + df['rightHip'][df['rightHip'].idxmax():].idxmin()) / 2)
        
    crop_df = df[['leftHip','rightHip']].iloc[lowest_level_1:lowest_level_2,:]
    
    return crop_df


def high_pass_filter(df):
    """
    フーリエ変換をして周波数領域に

    Returns:
        pd.DataFrame : フーリエ変換後のデータフレーム
    """
    fig = plt.figure(figsize=(24,9))
    ax = fig.add_subplot(1,2,1)
    ax.plot(df['leftHip'])
    ax.plot(df['rightHip'])


    df['fft_leftHip'] = np.fft.fft(df['leftHip'])
    df['fft_rightHip'] = np.fft.fft(df['rightHip'])
    print("df:",df)
    print("df.shape:",df.shape)

    
    ax = fig.add_subplot(1,2,2)
    freq = np.fft.fftfreq(df.shape[0])
    ax.plot(freq,df['fft_leftHip'].values.real)
    ax.plot(freq,df['fft_rightHip'].values.real)
    plt.show()

    return df

def cos_sim(v1, v2):
    """
    コサイン類似度を出す

    Returns:
        ndarray : コサイン類似度
    """    
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


"""
・項目
安定度(横へのブレ)
深さ(しっかりしゃがんでいるか)
負荷率(しゃがみ時のキツイ体勢を何秒維持出来ているか)
足の幅（足の幅は適切な距離離れているか=肩幅、少し広めはおけ）
太もも床平行（太ももが床と平行か）
膝（膝が内側に入りすぎていないか）

(秒数はタイミング合わせるの難しかったりリズム感無いと厳しいので保留)


・安定度
→横ブレの検知(比較先の動画のみにおいて、体幹に関するジャッジ)

・深さ
→角度を無視した場合(角度がつくとどれだけ深いか判定しにくい)。くるぶし:ひざ:腰の比率

・負荷率(しゃがみ時のキツイ体勢を何秒維持出来ているか?)

・足の幅
→leftankleとrightankleの最大距離(x)-leftshoulderとrightshoulderの最大距離(x)

・太もも床平行（太ももが床と平行か）
→righthipとrightkneeの最小距離(y)とlefthipとleftkneeの最小距離(y)の平均

・膝
→righthipとrightkneeの最小距離(y)の同時刻の時のleftankleとrightankleの距離(x)-righthipとrightkneeの最小距離(y)の同時刻の時のrightkneeとleftkneeの距離(x)
"""

def horizontal_stability(df):
    """
    ・安定度
    横ブレの検知をする。
    平均二乗誤差を出す
    比較元の安定度の1.0倍以下なら「とても良い」1.5倍以下なら「少しブレる」1.5倍以上なら「安定性が無い」

    Returns:
        float : 9箇所の平均二乗誤差
    """

    #ブレ (0~6,11~12: 頭、目、耳、肩、腰)
    nine_stability = 0
    for rep in [0,1,2,3,4,5,6,11,12]:
        nine_stability += ( df.iloc[:,rep].map(lambda x: np.power(x - df.iloc[:,0].mean(),2)).sum() / df.shape[0])
    return nine_stability/9  

def down_depth(df):
    """
    ・深さ
    身長に対してどれだけ沈んだか
    腰の高さの最大値と最小値の差を出す

    Returns:
        float : 腰の高さの最大値と最小値の差
    """

    #深さ: 腰の高さの最大値と最小値の差
    hip_depth = df.iloc[:,11].max() - df.iloc[:,11].min()
    
    return hip_depth

def feet_width(df):
    """
    ・足の幅
    leftankleとrightankleの最大距離(x)とleftshoulderとrightshoulderの最大距離(x)の差
    
    Returns:
        float : leftankleとrightankleの最大距離(x)とleftshoulderとrightshoulderの最大距離(x)の差
    """
    
    #leftankleとrightankleの最大距離(x)とleftshoulderとrightshoulderの最大距離(x)の差
    arrayankle = df.iloc[15,:] - df.iloc[16,:]
    arrayshoulder = df.iloc[5,:] - df.iloc[6,:]
    foot_width = np.max(arrayankle) - np.max(arrayshoulder)
    
    return foot_width
    
def thigh_parallel(df):
    """
    ・太もも床平行
    太ももが床と平行か
    
    Returns:
        float : righthipとrightkneeの最小距離(y)とlefthipとleftkneeの最小距離(y)の平均
    """
    
    #righthipとrightkneeの最小距離(y)とlefthipとleftkneeの最小距離(y)の平均
    arrayright = df.iloc[:,12] - df.iloc[:,14]
    arrayleft = df.iloc[:,11] - df.iloc[:,13]
    thigh_how = np.min(arrayright) + np.min(arrayleft) / 2
    
    return thigh_how
    
def knee_angle(df):
    """
    ・膝
    膝が内側に入りすぎていないか
    
    Returns:
        float : righthipとrightkneeの最小距離(y)の同時刻の時のleftankleとrightankleの距離(x)-righthipとrightkneeの最小距離(y)の同時刻の時のrightkneeとleftkneeの距離(x)
    """
        
    #righthipとrightkneeの最小距離(y)の同時刻の時のleftankleとrightankleの距離(x)-righthipとrightkneeの最小距離(y)の同時刻の時のrightkneeとleftkneeの距離(x)
    arrayfeet = df.iloc[15,:] - df.iloc[16,:]
    arrayknee = df.iloc[13,:] - df.iloc[14,:]
    knee_how = np.min(arrayfeet) - np.min(arrayknee)
    
    return knee_how
    
