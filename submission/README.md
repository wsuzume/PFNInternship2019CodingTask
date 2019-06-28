# Preferred Networks インターン選考2019コーディング課題 機械学習・数理分野

## 特記事項
時間的な都合により課題4について一部未解答（Adamの実装を試みましたが、中止して課題3までの実装を用いて `prediction.txt` を作成することを優先しました）。

## 実行環境
MacOS Mojave バージョン10.14.4
MacBook Air (13-inch, 2017)
プロセッサ 1.8GHz intel Core i5
メモリ 8GB 1600 MHz DDR3

Python anaconda3-5.0.1

## ディレクトリ構造
計算が重かったため`*.npy`に中間結果を書き出しています。これらのファイルは以下の`task2.py`から`task4.py`までを順に実行すれば生成されるため、不要であれば一度削除していただいて構いません。

* report.pdf ... 課題3,4のレポート
* prediction.txt ... 課題4でテストデータに対して予測したラベル
* src ... ソースコード
    * gnn.py ... 課題1,2,3,4を通したGNNの実装およびデータセットのパース処理を行うコード
    * test_task1.py ... 課題1のテストを実行するコード
    * task2.py ... 課題2を実行するコード
    * task3_sgd.py ... 課題3(SGD)を実行するコード
    * task3_momentum.py ... 課題3(MomentumSGD)を実行するコード
    * task3_measure.py ... 課題3の評価を実行するコード
    * task3_plot.ipynb ... レポートに記載した図を描画するノートブック
    * task4.py ... 課題4を実行するコード
    * loss.npy ... 課題2の損失減少過程
    * loss_sgd.npy ... 課題3(SGD)の損失減少過程
    * loss_momentum.npy ... 課題3(MomentumSGD)の損失減少過程
    * W_sgd.npy ... 課題3(SGD)で求めた重みW
    * A_sgd.npy ... 課題3(SGD)で求めた重みA
    * W_momentum.npy ... 課題3(MomentumSGD)で求めた重みW
    * A_momentum.npy ... 課題3(MomentumSGD)で求めた重みA

## gnn.pyについて
`gnn.py`は課題1〜4を通して用いるGNNクラスのコードと、課題3以降で用いるデータセットの読取りを行うコードが記述されています。GNNクラスについてはコード中のどこがどの課題に対応するかコメントで示してあります。

このソースコードを最初から使用するため、課題1,課題2でテストに用いるデータも課題3以降で使用するデータセットから隣接行列を抜粋しています。

## コードの実行方法
こちらの手元での計算結果が再現できるよう `gnn.py ` の冒頭で `np.random.seed(0)` を実行しています。必要に応じて seed の初期化をコメントアウトしてください。

ただし、時間の都合により、課題3でのSGDとMomentumSGDの評価の公平性を担保する根拠がこの乱数の初期化によるものとなってしまったため、`task3_*.py` を実行するときはこの初期化をなるべくコメントアウトしないでください。

### 課題1
課題1のコードのテストはPython標準のユニットテストモジュールを使用して実行できます。

```
python -m unittest ./src/test_task1.py
```

### 課題2,3,4
課題2,3,4のコードはいずれも stdout へ検証用の情報を表示し、matplotlibで可視化を行います。ただしレポートに記載した図は Jupyter Notebook  の`task3_plot.ipynb`で描画したものです。

`task3_plot.ipynb`以外は、以下のコマンドを用いて実行できます。`task4.py`は同ディレクトリ内に`prediction.txt`を生成することに注意してください。

```
python task2.py
python task3_sgd.py
python task3_momentum.py
python task3_measure.py
python task4.py
```
