# RNA二次構造予測の精度改善に向けた修士研究

## ファイル・ディレクトリ構造

mystudy/  
    ホームディレクトリ．ここでscriptsを実行．

- pretrain/ : 事前学習モデル．Editable Installを前提としている
    - models.py : MLMとdata2vec，あるいはその両方を組み合わせたモデルの実装．
    - train.py : 事前学習モデルの訓練（if main で単体でも実行できるように）．
    - test.py : 特徴表現の抽出．
    - module.py : modelで使用するレイヤやPEなどの実装．
    - utils.py : その他使用する関数の定義．
    - dataset.py : pytorchデータセットおよびその周辺の関数の定義．
    - conf/ : hydraを用いたパラメータ設定ファイルを含むディレクトリ．  
        - config.yaml : hydraを用いたパラメータ設定ファイル．
        - config.py : パラメータの型ヒントを与えるためのdataclass．
        - framework/ : mlmまたはdata2vecの設定を含むディレクトリ．  
            - mlm.yaml : mlmのパラメータ設定．
            - data2vec.yaml : data2vecのパラメータ設定．
        - model_size/ : モデルサイズごとのパラメータ設定．  
            - s.yaml : smallサイズ
            - m.yaml : midiumサイズ
            - l.yaml : largeサイズ
            - r.yaml : RiNALMoと同じサイズ

- SSpredictor/ : 二次構造予測モデル．（あるいはそれ以外の予測にも拡張可能性を残す）
    - model.py : 線形層のみ，MXfold2，その他既存モデルを組み込めるようなモデルフレームワークの実装．
    - train.py : モデルの学習．
    - test.py : 二次構造予測．
    - utils.py : その他使用する関数の定義．
    - dataset.py : pytorchデータセットおよびその周辺の関数の定義．
    - conf/ : hydraを用いたパラメータ設定ファイルを含むディレクトリ．  
        - config.yaml : hydraを用いたパラメータ設定ファイル．
        - config.py : パラメータの型ヒントを与えるためのdataclass．

- scripts/ : pretrainやpredictorの実行．
    - pretrain.py : 事前学習モデルの訓練，特徴表現の抽出，あるいはその両方を一括で実行．
    - archiveii_kfold.py :  

- notebook/ : 結果に対する分析や解析．

- data/ : 事前学習データや二次構造予測データの保管．  
    - embeddings/ : 二次構造予測に使われる特徴表現．
    - pretrain_data/ : 事前学習モデルの訓練（と検証）に使うデータ．
    - SS_data/ : 二次構造予測に使うデータ．

- results/ : 事前学習や二次構造予測の結果の保管．  
    - pretrain_results/ : 事前学習の結果（重み，特徴表現など）を保管．
    - SS_results/ : 二次構造予測の結果を保管．

- .devcontainer/ : VScode用の開発コンテナ．
    - devcontainer.json : コンテナ設定ファイル．
- Dockerfile : 開発コンテナでも普通のコンテナでも使えるDockerfile．
- environment.yaml : Dockerコンテナにインストールする環境の設定
- README.md : これ