import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_graph(x_df,y_df):
    x_df = x_df[['nose','leftEye','rightEye','leftEar','rightEar','leftShoulder','rightShoulder','leftElbow','rightElbow','leftWrist','rightWrist','leftHip','rightHip','leftKnee','rightKnee','leftAnkle','rightAnkle']]
    y_df = y_df[['nose','leftEye','rightEye','leftEar','rightEar','leftShoulder','rightShoulder','leftElbow','rightElbow','leftWrist','rightWrist','leftHip','rightHip','leftKnee','rightKnee','leftAnkle','rightAnkle']]
    
    fig  = plt.figure(figsize=(24,9))

    #x方向の移動
    ax_x = fig.add_subplot(2,2,1)
    for i in range(x_df.shape[1]):
        ax_x.plot(x_df.iloc[:,i])
    ax_x.set_xlabel('time (s)')
    ax_x.set_ylabel('movement distance (pixel)')
    ax_x.set_title('movement forward x axis')
    ax_x.legend(
        ['nose','leftEye','rightEye','leftEar','rightEar','leftShoulder','rightShoulder','leftElbow','rightElbow','leftWrist','rightWrist','leftHip','rightHip','leftKnee','rightKnee','leftAnkle','rightAnkle'],
        loc='upper left', borderaxespad=0, fontsize=8
        )

    #y方向の移動
    ax_y = fig.add_subplot(2,2,2)
    for i in range(y_df.shape[1]):
        ax_y.plot(y_df.iloc[:,i])
    ax_y.set_xlabel('time (s)')
    ax_y.set_ylabel('movement distance (pixel)')
    ax_y.set_title('movement forward y axis')
    ax_y.legend(
        ['nose','leftEye','rightEye','leftEar','rightEar','leftShoulder','rightShoulder','leftElbow','rightElbow','leftWrist','rightWrist','leftHip','rightHip','leftKnee','rightKnee','leftAnkle','rightAnkle'],
        loc='upper left', borderaxespad=0, fontsize=8
        )

    #x方向の移動(主要点)
    ax_x = fig.add_subplot(2,2,3)
    for clm in ['leftElbow','rightElbow','leftWrist','rightWrist']:
        ax_x.plot(x_df.loc[:,clm])
    ax_x.set_xlabel('time (s)')
    ax_x.set_ylabel('movement distance (pixel)')
    ax_x.set_title('movement of main points forward x axis')
    ax_x.legend(
        ['leftElbow','rightElbow','leftWrist','rightWrist'],
        loc='upper left', borderaxespad=0, fontsize=8
        )
    
    #y方向の移動(主要点)
    ax_y = fig.add_subplot(2,2,4)
    for clm in ['nose','leftEye','rightEye','leftEar','rightEar','leftShoulder','rightShoulder','leftElbow','rightElbow','leftWrist','rightWrist','leftHip','rightHip','leftKnee','rightKnee']:
        ax_y.plot(y_df.loc[:,clm])
    ax_y.set_xlabel('time (s)')
    ax_y.set_ylabel('movement distance (pixel)')
    ax_y.set_title('movement of main points forward y axis')
    ax_y.legend(
        ['nose','leftEye','rightEye','leftEar','rightEar','leftShoulder','rightShoulder','leftElbow','rightElbow','leftWrist','rightWrist','leftHip','rightHip','leftKnee','rightKnee'], 
        loc='upper left', borderaxespad=0, fontsize=8
        )

    plt.show()

def mark_score(x_df,y_df):
    x_df['second'] = x_df.index/x_df.iloc[0,x_df.shape[1]-1]
    print(x_df['second'])

if __name__ == "__main__":
    squat1_x_df = pd.read_csv(f'../../images/squat1_x_df.csv',index_col=0)
    squat1_y_df = pd.read_csv(f'../../images/squat1_y_df.csv',index_col=0)
    make_graph(squat1_x_df,squat1_y_df)
    squat2_x_df = pd.read_csv(f'../../images/squat2_x_df.csv',index_col=0)
    squat2_y_df = pd.read_csv(f'../../images/squat2_y_df.csv',index_col=0)
    make_graph(squat2_x_df,squat2_y_df)
    mark_score(squat1_x_df,squat1_y_df)
    mark_score(squat2_x_df,squat2_y_df)
    print(squat1_x_df.shape)