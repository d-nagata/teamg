import tensorflow as tf
import cv2
import time
import argparse

import posenet

#ここから自分でいじったパート
import pandas as pd
"""
(注意)
現段階では急を要するのでwebcam_demo.pyに全てを纏めて書いていますが、
リファクタリングする必要があります。
"""


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

"""
実際に動画を読み込み座標を特定、画像にオーバーライトする関数
mainとなっているのにmain関数ではないので、命名も変える必要あり
"""
def main():
    try:
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(args.model, sess)
            output_stride = model_cfg['output_stride']

            if args.file is not None:
                cap = cv2.VideoCapture(args.file)
            else:
                cap = cv2.VideoCapture(args.cam_id)
            cap.set(3, args.cam_width)
            cap.set(4, args.cam_height)

            #動画の長さ
            video_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
            video_fps = cap.get(cv2.CAP_PROP_FPS)           # FPS を取得する
            video_len_sec = video_frame / video_fps                     # 長さ（秒）を計算する

            start = time.time()
            frame_count = 0
            #インクリメント用
            cnt = 0
            #ファイルをxy軸で追加
            datalist_x = []
            datalist_y = []
            while True:
                #インクリメントして途中で中断
                cnt+= 1
                if cnt >20:
                    pass
                    #break

                input_image, display_image, output_scale = posenet.read_cap(
                    cap, scale_factor=args.scale_factor, output_stride=output_stride)

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    model_outputs,
                    feed_dict={'image:0': input_image}
                )

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0),
                    offsets_result.squeeze(axis=0),
                    displacement_fwd_result.squeeze(axis=0),
                    displacement_bwd_result.squeeze(axis=0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)

                keypoint_coords *= output_scale
                
                #手首表示
                '''
                rtwrt[1]がx座標,rtwrt[0]がy座標になってます。
                [0,10,:]の”0”が一人目という意味で”10”というのが右手首(rightWrist)という意味です。
                全部で17か所あって,配列の0～16にはそれぞれ

                nose','leftEye','rightEye','leftEar','rightEar','leftShoulder',
                'rightShoulder','leftElbow','rightElbow','leftWrist','rightWrist',
                'leftHip','rightHip','leftKnee','rightKnee','leftAnkle','rightAnkle'
                
                となってます。

                '''
                datalist_x.append(keypoint_coords[0,:,1])
                datalist_y.append(keypoint_coords[0,:,0])

                # TODO this isn't particularly fast, use GL for drawing and display someday...
                overlay_image = posenet.draw_skel_and_kp(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.15, min_part_score=0.1)

                cv2.imshow('posenet', overlay_image)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print('Average FPS: ', frame_count / (time.time() - start))

    except:
        #OSError: webcam failureが出るので、対策
        print('Average FPS: ', frame_count / (time.time() - start))

    return datalist_x,datalist_y,video_len_sec


"""
とりあえずcsv保存,グラフ描画
"""
def save_as_csv(datalist_x,datalist_y,video_len_sec):
    #動画の長さも無理やり保存
    x_df = pd.DataFrame(datalist_x,columns=['nose','leftEye','rightEye','leftEar','rightEar','leftShoulder','rightShoulder','leftElbow','rightElbow','leftWrist','rightWrist','leftHip','rightHip','leftKnee','rightKnee','leftAnkle','rightAnkle'])
    x_df['length'] = 0
    x_df.iloc[0,len(x_df.columns)-1] = [video_len_sec]
    y_df = pd.DataFrame(datalist_y,columns=['nose','leftEye','rightEye','leftEar','rightEar','leftShoulder','rightShoulder','leftElbow','rightElbow','leftWrist','rightWrist','leftHip','rightHip','leftKnee','rightKnee','leftAnkle','rightAnkle'])
    y_df['length'] = 0
    y_df.iloc[0,len(y_df.columns)-1] = [video_len_sec]


    #とりあえず保存しておく
    filename = args.file
    filename = filename.split('.')[0]
    x_df.to_csv(f'../../images/{filename}_x_df.csv')
    y_df.to_csv(f'../../images/{filename}_y_df.csv')


if __name__ == "__main__":
    datalist_x,datalist_y,video_len_sec = main()
    save_as_csv(datalist_x,datalist_y,video_len_sec)
    