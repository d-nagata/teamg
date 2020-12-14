# teamg 筋トレアプリ(仮)

***

## ディレクトリ説明

### images

動画を読み込んだ際に生成されたデータフレームをcsv形式で保存しています。  
命名方法は {動画の名前}_{x,y座標のどちらか}_df.csv のようになっています。  
ex) squat1_x_df.csv　→　squat1 動画の x 座標のデータフレームをcsv化したもの  

### Scripts

ここに基本的にpyファイルやposenet-pythonが入っていたりします。

- *.mp4: 
    - 動画を入れています。今回使ったデモ動画は squat1.mp4,squat2.mp4 です。
- modules.py: 
    - データ分析に用いる関数をまとめています。  現在 「読み込み→標準化→(グラフ描画)→安定度、深さを出す」までを実行しています。
- conduct.py:
    - modules.pyにある関数を実行します。
- fig_graph.ipynb:
    - 残骸です。
- posenet-python-master:
    - webcam_demo.pyのみ動かせれば問題無いです。使用方法は、「python3 posenet-python-master/webcam_demo.py --file squat2.mp4」のように打ち込んでください(Scripts)  詳しくは[github](https://github.com/rwightman/posenet-python)に載っています。

### 問題点
- webcam_demo.pyでsquat2の読み込みが途中で終わってしまう。
- 動画の秒数が取得できれば負荷率が出せたり幅が広がる→ffmpegというモジュールを使えばなんとかなる？
