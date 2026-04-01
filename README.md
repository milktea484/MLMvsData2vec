# RNA二次構造予測の精度改善に向けた修士研究

## ファイル・ディレクトリ構造

mystudy/  
    ホームディレクトリ．ここでscriptsを実行．

- pretrain/ : 事前学習モデル．Editable Installを前提としている
    - models.py : MLMとdata2vec，あるいはその両方を組み合わせたモデルの実装．
    - train.py : 事前学習モデルの訓練．
    - test.py : 特徴表現の抽出．
    - module.py : modelで使用するレイヤやPEなどの実装．
    - utils.py : その他使用する関数の定義．
    - dataset.py : pytorchデータセットおよびその周辺の関数の定義．
    - pyproject.toml : Editable Install用のtomlファイル．
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
        - lr_scheduler/ : 学習率スケジューラのパラメータ設定．
            - cosine.yaml : コサインスケジューラ
        - optimizer/ : オプティマイザーのパラメータ設定．
            - adamw.yaml : AdamW

- SSpredictor/ : 二次構造予測モデル．（あるいはそれ以外の予測にも拡張可能性を残す）
    - models.py : 線形層のみ，MXfold2，その他既存モデルを組み込めるようなモデルフレームワークの実装．
    - modules.py : モデル構築や学習率スケジューラなどを定義．
    - train.py : モデルの学習．
    - test.py : 二次構造予測．
    - utils.py : その他使用する関数の定義．
    - dataset.py : pytorchデータセットおよびその周辺の関数の定義．
    - knotfold/ : KnotFoldのmin cost flowアルゴリズム
        - KnotFold_mincostflow.cc
        - KnotFold_mincostflow
    - conf/ : hydraを用いたパラメータ設定ファイルを含むディレクトリ．  
        - config.yaml : hydraを用いたパラメータ設定ファイル．
        - config.py : パラメータの型ヒントを与えるためのdataclass．
        - model/ : モデルのパラメータ設定
            - knotfold.yaml : KnotFold
        - optimizer/ : オプティマイザーのパラメータ設定
            - adamw.yaml : AdamW

- scripts/ : pretrainやpredictorの実行．
    - pretrain.py : 事前学習モデルの訓練，特徴表現の抽出，あるいはその両方を一括で実行．
    - archiveii_kfold.py :  
    - run_archiveii_famfold.py : ArchiveII の fam-fold スキームで SSpredictor/train.py・test.py を family ごとに実行し，結果を集約するラッパスクリプト．

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

## scripts/run_archiveii_famfold.py 実行時の処理フローと注意点

### 全体フロー概要

- 入力データ
    - data/SS_data/ArchiveII.csv, data/SS_data/ArchiveII_famfold_splits.csv を読み込み，`id` 列をキーに family ごとの train/test を切り出す．
    - family ごとに data/SS_data/archiveii_famfold/{fam}/train.csv, test.csv を生成する．
- 学習・評価の実行
    - 各 family について，SSpredictor/train.py を Hydra 設定 (experiment.name=ArchiveII_famfold, experiment.additional_experiment_info={fam} など) 付きで実行し，KnotFold モデルを学習する．
    - 続いて，同じ family に対して SSpredictor/test.py を実行し，学習済みモデルでテスト・評価・予測を行う．
- 結果の集約
    - family ごとに生成された prediction_results.csv を results/SS_results/ArchiveII_famfold/.../overall_results.csv にまとめる．

### データ準備と split に関する注意

- ArchiveII.csv と ArchiveII_famfold_splits.csv の id は一致している前提であり，いずれか一方にのみ存在する id は単に無視される (エラーにはならないが，意図せぬサンプル欠落の原因になり得る)．
- ある family について train または test が 0 行になると，下流で以下の問題が起こり得る:
    - train_loader の長さが 0 になり，SSpredictor/train.py 内で total_steps=0 となる．この場合，CosineScheduler の内部計算で 0 除算が発生する可能性がある．
    - 学習ループが 1 回も回らず，最良モデルが保存されないと，最後の assert (best_model_state_dict is not None) で停止する．
    - val/test 用 DataLoader が空のときは next(iter(loader)) で StopIteration が発生しうる．

### pretrain モデル / 埋め込みと Tensor まわりの注意

- SSpredictor/train.py は utils.validate_config により，以下のどちらかが必須:
    - pretrain.framework と pretrain.timestamp を指定し，pretrain 側の BaseModel._test から埋め込みを得る．
    - dataset.embedding_file を指定し，data/embeddings/ 以下の HDF5 から埋め込みを読む．
- どちらも指定しないと train 開始前に ValueError で停止するが，指定した場合でも次が守られている必要がある:
    - pretrain_model_dir/framework/timestamp 配下に train_config/.hydra/config.yaml と weight_*.pth (もしくは teacher_weight_*.pth) が存在すること．
    - 埋め込み HDF5 内に，全シーケンス id に対応するデータセットが存在し，shape が一貫していること (attention なら (E,L,L)，そうでなければ (L,E))．
- dataset.EmbeddingDataset / pad_batch:
    - CSV 側には id, sequence, base_pairs 列が必要で，base_pairs は JSON 文字列としてパース可能である前提．
    - bp_matrix は int8 の {-1,0,1} で保持され，KnotFoldModel.loss_func 内で -1 は weight=0 にすることで BCE の無視ラベルとして扱われる．
    - embeddings / reference_embeddings は float32 で，train/test ループでは torch.autocast(bfloat16) のコンテキスト内で使われるため，dtype 変換は PyTorch 側に任せている．
    - dataset が空 (len(loader)==0) の場合，get_embedding_dim(loader, use_attention) の next(iter(loader)) で StopIteration が投げられる点に注意．

### test.py (評価・予測) に関する注意

- test.py はまず学習時の Hydra 設定 (train_config/config.yaml) を読み込み，そこから train_cfg を復元した上で，dataset.test_file や common.iterations などを上書きする．
- train_cfg.common.iterations 以上の iterations は指定できないようにチェックされており，かつ 1 以上である必要がある (それ以外は ValueError)．
- 重みロード:
    - 各 iteration ごとに weights/prior_<iteration>.pth (および KnotFold の場合 weights/reference_<iteration>.pth) を torch.load するため，学習時にその回数分の重みが保存されていないと FileNotFoundError になる．
- 評価・AUC 計算:
    - 確率行列 pred_bp_prob (torch.Tensor) を .cpu().numpy() に変換してから metrics.calculate_confusion_matrix および calculate_auc に渡しており，GPU 実行時も CPU に明示的に移して評価している．
    - test_loader が 1 バッチも返さない場合，overall_results["test_losses"] の平均が NaN になり，混同行列に基づく AUC も NaN を含む可能性がある (エラーではなく数値の問題として現れる)．
- 予測と KnotFold:
    - cfg.common.prediction が True のとき，各配列の (gt_bp_matrix, pred_bp_prob, ref_bp_prob) を CPU numpy 配列にして一時ファイル (prior.mat, reference.mat) に書き出し，knotfold/KnotFold_mincostflow を subprocess で呼び出す．
    - 実行コマンドの戻り値が 0 でない場合は assert p.returncode == 0 により即座に停止する．したがって，実行ファイルの存在・実行権限・入出力フォーマットが Docker 環境で正しく整っている必要がある．

### まとめ (run_archiveii_famfold 実行時に特に注意すべき点)

- family ごとに train/test に 1 サンプル以上存在するように ArchiveII_famfold_splits.csv を設計すること．
- ArchiveII.csv・splits・埋め込み HDF5 の id と shape を事前に検証し，不整合がないことを確認すること．
- pretrain モデルのディレクトリ構造・重みファイル・config.yaml が，SSpredictor の想定どおりに存在していること．
- KnotFold_mincostflow がコンテナ内で実行可能であり，期待どおりの行列フォーマットで入出力していること．

これらの前提を満たせば，現在の実装 (train.py/test.py/dataset.py/models.py など) において，torch.Tensor や np.ndarray の扱い，pretrain モデルとの連携に起因する致命的な構造バグは確認されていません。