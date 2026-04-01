# SSpredictor

二次構造予測モデル (KnotFold) の学習・評価用スクリプト群です。ここでは、とくに `train.py` と `test.py` を直接/間接に実行するときに注意すべき点をまとめます。

- 設定管理は Hydra (`conf/config.yaml`, `conf/test_config.yaml`) を前提としています。
- 実際の運用では、上位スクリプト (例: `scripts/run_archiveii_famfold.py`) から呼ばれることを想定していますが、単体実行時も同じ前提条件が必要です。

---

## 1. `train.py` 実行時の注意点

### 1-1. 役割

- Hydra 設定 (`conf/config.yaml`) に基づいて、二次構造予測モデル (現在は KnotFold) を学習し、最良モデルの重みを保存します。
- 事前学習モデル (`pretrain` パッケージ) またはあらかじめ計算済みの埋め込み (`dataset.embedding_file`) のどちらかを用いて特徴量を構成します。

### 1-2. 設定ファイルと必須パラメータ

- 使用する設定ファイル: `conf/config.yaml`
- 重要パラメータ (抜粋):
  - `experiment.name`
    - 既定値は `"???"` なので、必ず実験名に上書きしてください。
    - データパス・結果パスの両方に使用されます。
  - `pretrain.framework`, `pretrain.timestamp`, `dataset.embedding_file`
    - `validate_config` により、以下のいずれかが必須です。
      - (a) `pretrain.framework` と `pretrain.timestamp` を指定して事前学習モデルを使う
      - (b) `dataset.embedding_file` を指定して HDF5 埋め込みを使う
    - どちらも指定しない場合、実行開始時にエラーになります。
  - `path` 関連
    - `path.data_dir` : データルートディレクトリ (既定: `./data/SS_data/`)
    - `path.embedding_dir` : 埋め込み HDF5 のディレクトリ (既定: `./data/embeddings/`)
    - `path.pretrain_model_dir` : 事前学習モデルの保存ディレクトリ (既定: `./results/pretrain_results/`)
    - `path.output_dir` : SS 予測モデルの結果ルート (既定: `./results/SS_results/${experiment.name}/`)
    - `path.timestamp` : 学習実行ごとのタイムスタンプ。Hydra により自動で付与されます。
  - `dataset` 関連
    - `dataset.sequence_file` : 元データ CSV ファイル名 (例: `ArchiveII.csv`)
    - `dataset.train_file`, `dataset.validation_file`, `dataset.test_file` : 各 split の CSV ファイル名
    - それぞれの CSV は `id`, `sequence`, `base_pairs` の列を必ず持つ必要があります。
  - `common` 関連
    - `batch_size > 0`, `max_epochs > 0`, `eval_per_epoch >= 1`, `eval_steps > 0`, `iterations >= 1` になるように設定してください。
    - これらが 0 や負の値の場合、学習ループやスケジューラの挙動が破綻し、実行時エラーになる可能性があります。

### 1-3. データパスとファイル構成

- 実際に読み込まれる CSV のパスは `dataset.py` の `create_dataloader` により決まります。
  - `split` ごとのファイル名は `config.dataset.*_file` で指定されます。
  - パスの構成:
    - `config.experiment.additional_experiment_info` が **未指定** の場合:
      - `${path.data_dir}/${experiment.name.lower()}/${dataset.<split>_file}`
    - `config.experiment.additional_experiment_info` を **指定**した場合:
      - `${path.data_dir}/${experiment.name.lower()}/${experiment.additional_experiment_info}/${dataset.<split>_file}`
- したがって、以下を満たす必要があります。
  - `path.data_dir` の下に、`experiment.name.lower()` と同名のサブディレクトリが存在すること。
  - `additional_experiment_info` を使う場合、そのサブディレクトリ配下に split ごとの CSV が存在すること。

### 1-4. 事前学習モデル / 埋め込み周りの前提

- 事前学習モデルを利用する場合:
  - モデルパス: `${path.pretrain_model_dir}/${pretrain.framework}/${pretrain.timestamp}/`
  - 上記ディレクトリに、少なくとも以下が存在している必要があります。
    - `train_config/.hydra/config.yaml` : 事前学習モデルの Hydra 設定
    - `weight_<checkpoint>.pth` もしくは `teacher_weight_<checkpoint>.pth`
  - `cfg.pretrain.checkpoint == "final"` の場合は `pretrain_cfg.common.max_steps` が使われ、それ以外は指定値がそのままチェックポイント番号として使われます。
- `dataset.embedding_file` を使う場合:
  - パス: `${path.embedding_dir}/${dataset.embedding_file}`
  - HDF5 内には、学習に使う全シーケンス `id` に対応するデータセットが存在している必要があります。
  - attention 形式の埋め込み (`(E, L, L)` など) と通常の埋め込み (`(L, E)`) の両方に対応していますが、どちらかに統一されている前提です。

### 1-5. 学習と出力、よくあるエラー要因

- 出力ディレクトリ:
  - 実際の保存先は次のような構造になります。
    - `${path.output_dir}/` 以下に、
      - 事前学習モデルあり: `${pretrain.framework}/${pretrain.timestamp}/${model.name}/${path.timestamp}/[${experiment.additional_experiment_info}/]`
      - 埋め込みのみ: `${embedding_name}/${model.name}/${path.timestamp}/[${experiment.additional_experiment_info}/]`
  - このディレクトリの直下に `log_train.txt`, `weights/prior_<iteration>.pth`, (KnotFold の場合 `weights/reference_<iteration>.pth`) などが保存されます。
- よくあるエラー・注意点 (コード構造起因のもの):
  - `pretrain.framework` / `pretrain.timestamp` と `dataset.embedding_file` をすべて未指定にして実行すると、埋め込み次元が決定できず、`assert` で停止します。
  - `experiment.name` や `additional_experiment_info` を誤って指定すると、`create_dataloader` が指す CSV パスが存在せず、`FileNotFoundError` になります。
  - 学習データが空、あるいは極端に少ない場合、`len(train_loader) == 0` などに起因して学習ループやスケジューラが正常に回らない可能性があります。
  - KnotFold モデル使用時は `split="reference"` 用のデータも必要であり、`create_dataloader(config, split="reference", ...)` が参照する CSV が存在している必要があります。

---

## 2. `test.py` 実行時の注意点

### 2-1. 役割

- 学習済みモデルの重みを読み込み、テストデータに対する損失・ROC/PR AUC などを計算し、必要に応じて予測結果 (二次構造ペア) を CSV として出力します。
- KnotFold の場合、外部バイナリ (KnotFold_mincostflow) を用いて `kf_lambda` のチューニングと最終予測を行います。

### 2-2. 設定ファイルと必須パラメータ

- 使用する設定ファイル: `conf/test_config.yaml`
- 重要パラメータ (抜粋):
  - `experiment.name`
    - 学習時と同じ実験名を指定する必要があります。
  - `SStrain_model_path.model_name`, `SStrain_model_path.timestamp`
    - 学習時の `cfg.model.name`, `cfg.path.timestamp` と一致させてください。
    - `train_model_path` は以下のように構築されます。
      - ベース: `cfg.path.output_dir` (既定: `./results/SS_results/${experiment.name}/`)
      - 事前学習モデルあり: `/<pretrain.framework>/<pretrain.timestamp>/`
      - 埋め込みのみ: `/embedding_name/`
      - その下に `/<SStrain_model_path.model_name>/<SStrain_model_path.timestamp>/[<experiment.additional_experiment_info>/]`
    - ここが学習時の出力ディレクトリと一致していないと、`FileNotFoundError` になります。
  - `pretrain.framework`, `pretrain.timestamp`
    - 明示的に指定しない場合でも、`train_config` 側の設定から事前学習モデルの情報を復元します。
    - 事前学習モデルを用いた学習を行っている場合は、そのディレクトリ構成と重みファイルが学習時と同じ場所に存在している必要があります。
  - `dataset.test_file`
    - 既定値は `test.csv` です。
    - `train.py` 実行時と同様に、実際のパスは `train_config` の `path.data_dir`, `experiment.name.lower()`, `experiment.additional_experiment_info` を使って組み立てられます。
    - `SSpredictor/test.py` はいったん `train_config` を読み込んだ後で `train_cfg.dataset.test_file` を `cfg.dataset.test_file` で上書きし、そのファイル名を用いてテスト用 DataLoader を作ります。
  - `common.iterations`
    - 省略時 (`null`) は学習時の `train_cfg.common.iterations` がそのまま使われます。
    - 明示的に指定した場合、`cfg.common.iterations` は `train_cfg.common.iterations` 以下の正の整数である必要があります。
      - これを満たさないと `ValueError` になります。
  - `experiment.kf_lambda_cfg` (KnotFold のみ)
    - `min`, `max`, `step` をすべて指定した場合、その範囲で `kf_lambda` を走査します。
    - いずれかが未指定の場合、学習時の `kf_lambda` (単一値) を用います。

### 2-3. 学習設定の復元と依存関係

- `test.py` は、まず学習時の Hydra 設定を読み込みます。
  - パス: `${cfg.path.output_dir}/${SStrain_model_path.timestamp}/train_config/config.yaml`
  - ここで読み込まれた `train_cfg` をベースに、`dataset.test_file` や `common.iterations` を上書きしてテスト用設定を構成します。
- 事前学習モデルを使用している場合:
  - `train_cfg.pretrain.framework`, `train_cfg.pretrain.timestamp`, `train_cfg.path.pretrain_model_dir` から事前学習モデルの場所を決定します。
  - ここで指定されるディレクトリ構成と重みファイル (Train のときと同じ) が存在していないと、`FileNotFoundError` や `torch.load` のエラーになります。

### 2-4. 重みファイルと反復処理

- `common.iterations` 回数だけ、学習済みモデルを読み込んでテストを行います。
  - 期待される重みファイル:
    - メインモデル: `weights/prior_<iteration>.pth` (0 始まり)
    - KnotFold の参照モデル: `weights/reference_<iteration>.pth`
  - いずれかが欠けていると、`torch.load` の段階でエラーになります。
- テスト処理:
  - 各 iteration ごとに `_test` を実行し、`loss` と `logits` を集計します。
  - 最終的には iteration 方向に平均した `mean_logits`, `mean_test_loss` を用いて評価します。
- `cfg.common.save_probability_matrix == true` の場合:
  - `test_results/<timestamp>/predicted_probability_matrices.h5` に全シーケンスの推定確率行列が保存されます。
  - 全バッチを HDF5 に書き出すため、データセットと `batch_size` の設定によってはファイルサイズが大きくなります。

### 2-5. 評価・予測・出力ファイル

- 評価 (`cfg.common.evaluation == true`):
  - 混同行列から ROC AUC, PR AUC を計算し、`auc_curve.png` として保存します。
  - これらの値は `overall_results.csv` にも記録されます。
- 予測 (`cfg.common.prediction == true`):
  - `model.predict` に対して、全バッチを `batch_list` として渡して最終予測を行います。
    - メモリにすべてのバッチを貯める実装のため、大規模データセットを扱う場合はメモリ消費に注意してください。
  - 出力ファイルの例:
    - `prediction_results.csv` : 各配列について `id`, `sequence`, 予測された塩基対 `base_pairs`, 長さ `len`, precision/recall/F1 などを保存
    - `kf_lambda_results.csv` : 各 `kf_lambda` に対する評価指標 (KnotFold のみ)
    - `overall_results.csv` : テスト損失・(あれば) 参照モデルの損失・ROC/PR AUC 等

---

## 3. train/test 共通の実行例

ここでは、リポジトリルート (`mystudy`) からの単体実行例を示します (Hydra を直接使う場合)。実際には、上位スクリプトから同等の引数が渡されます。

```bash
# 学習 (事前学習モデルを使わず、埋め込みファイルを指定する例)
python SSpredictor/train.py \
  experiment.name=ArchiveII \
  experiment.additional_experiment_info=familyA \
  dataset.embedding_file=ArchiveII_embeddings.h5 \
  dataset.train_file=train_familyA.csv \
  dataset.validation_file=val_familyA.csv

# テスト (学習済みモデルを指定して評価 + 予測を行う例)
python SSpredictor/test.py \
  experiment.name=ArchiveII \
  experiment.additional_experiment_info=familyA \
  SStrain_model_path.model_name=knotfold \
  SStrain_model_path.timestamp=20260204T163812 \
  dataset.test_file=test_familyA.csv
```

上記はあくまで一例です。実際には `scripts/run_archiveii_famfold.py` などのラッパースクリプトから、自動的に `experiment.name`, `additional_experiment_info`, `dataset.*_file`, `pretrain.*` などが設定されます。単体で実行する場合も、本 README に挙げた前提条件・ディレクトリ構成を満たすように設定してください。
