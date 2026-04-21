# scripts ディレクトリのスクリプト概要

このディレクトリには、事前学習および二次構造予測用のラッパースクリプトが含まれます。

## run_archiveii_famfold.py の典型的な実行例

以下は、ArchiveII_famfold 実験を行う際の典型的なコマンド例です．

1. 事前学習モデルを用いる場合  
    1. 事前学習モデルが1つの場合（data2vec、最終 checkpoint、simple アーキテクチャ）  
        ```bash
        python scripts/run_archiveii_famfold.py \
          'train:pretrain.framework=data2vec' \
          'train:pretrain.timestamp=20260307T153057' \
          'train:pretrain.checkpoint=final' \
          'train:model.arch.use_simple=true'
        ```

        一行バージョン
        ```bash
        python scripts/run_archiveii_famfold.py 'train:pretrain.framework=data2vec' 'train:pretrain.timestamp=20260406T124643' 'train:pretrain.checkpoint=final' 'train:model.arch.use_simple=true'
        ```

    2. 事前学習モデルを複数用いる場合（data2vec, mlm、最終 checkpoint、simple アーキテクチャ）  
        ```pretrain.framework``` と ```pretrain.timestamp``` をリストで記入（空白は無し）．  
        リスト内の順序に注意すること．また，表現の次元が一致している必要あり．

        ```bash
        python scripts/run_archiveii_famfold.py \
          'train:pretrain.framework=["data2vec","mlm"]' \
          'train:pretrain.timestamp=["20260307T153057","20260316T030756"]' \
          'train:pretrain.checkpoint=final' \
          'train:model.arch.use_simple=true'
        ```

        一行バージョン
        ```bash
        python scripts/run_archiveii_famfold.py 'train:pretrain.framework=["data2vec","mlm"]' 'train:pretrain.timestamp=["20260307T153057","20260316T030756"]' 'train:pretrain.checkpoint=final' 'train:model.arch.use_simple=true'
        ```


2. 事前計算済みの埋め込みファイルを用いる場合  
    1. 埋め込みファイルが1つの場合（simple アーキテクチャ）
        ```bash
        python scripts/run_archiveii_famfold.py \
          'train:dataset.embedding_file=ArchiveII_data2vec.h5' \
          'train:model.arch.use_simple=true'
        ```

        一行バージョン
        ```bash
        python scripts/run_archiveii_famfold.py 'train:dataset.embedding_file=ArchiveII_data2vec.h5' 'train:model.arch.use_simple=true'
        ```

    2. 埋め込みファイルが複数の場合（simple アーキテクチャ）  
        ```dataset.embedding_file``` をリストで記入（空白は無し）．  
        表現の次元が一致している必要あり．

        ```bash
        python scripts/run_archiveii_famfold.py \
          'train:dataset.embedding_file=["ArchiveII_data2vec.h5","ArchiveII_mlm.h5"]' \
          'train:model.arch.use_simple=true'
        ```
        
        一行バージョン
        ```bash
        python scripts/run_archiveii_famfold.py 'train:dataset.embedding_file=["ArchiveII_data2vec.h5","ArchiveII_mlm.h5"]' 'train:model.arch.use_simple=true'
        ```

3. 事前学習モデルと埋め込みファイルの両方を用いる場合（simple アーキテクチャ）
    ```pretrain.framework, pretrain.timestamp``` や ```dataset.embedding_file``` が複数ある場合はリストで指定（空白は無し）．  
    双方の表現の次元が一致している必要あり．

    ```bash
    python scripts/run_archiveii_famfold.py \
      'train:pretrain.framework=data2vec' \
      'train:pretrain.timestamp=20260307T153057' \
      'train:pretrain.checkpoint=final' \
      'train:dataset.embedding_file=["ArchiveII_data2vec.h5","ArchiveII_mlm.h5"]' \
      'train:model.arch.use_simple=true'
    ```

補足:

- `train:model=` を明示的に指定しない場合、二次構造予測モデルは自動的に `model=knotfold` が選択され、その情報が test 側 (`SStrain_model_path.model_name`) にも引き継がれます。
- 上記の例では simple アーキテクチャ (`model.arch.use_simple=true`) を指定しており、KnotFold の中でも線形層のみの軽量な構成で学習・推論を行います。

- pretrain.py
  - pretrain パッケージを用いた事前学習や埋め込み抽出の一括実行用スクリプト。
- run_archiveii_famfold.py
  - SSpredictor パッケージの train.py / test.py を呼び出し、ArchiveII データセットの FamFold クロスバリデーションを一括実行するスクリプト。
- run_archiveii_kfold.py
  - ArchiveII_kfold 実験用スクリプト（現在は未実装）。

---

## run_archiveii_famfold.py

### 役割

- リポジトリ直下の data/SS_data/ArchiveII.csv と data/SS_data/ArchiveII_famfold_splits.csv を読み込み、fold（RNA family）ごとに train/test CSV を生成します。
  - 生成先: data/SS_data/archiveii_famfold/{family}/train.csv, test.csv
- 各 family ごとに SSpredictor/train.py, SSpredictor/test.py をサブプロセスとして実行し、学習および推論を行います。
- 最後に、各 family の prediction_results.csv を集約して overall_results.csv を作成します。

### 実行方法

- リポジトリルート（myprograming/mystudy）をカレントディレクトリにして実行することを推奨します。

例:

- 事前学習モデルを使う場合:

  python scripts/run_archiveii_famfold.py \
    "train:pretrain.framework=data2vec" \
    "train:pretrain.timestamp=20260316T130913" \
    "train:pretrain.checkpoint=final"

- 事前計算済み埋め込みを使う場合:

  python scripts/run_archiveii_famfold.py \
    "train:dataset.embedding_file=ArchiveII_data2vec.h5"

- 特定の family のみ実行する場合:

  python scripts/run_archiveii_famfold.py \
    "train:pretrain.framework=data2vec" \
    "train:pretrain.timestamp=20260316T130913" \
    "family=RF00001"

### 必須ファイル・前提条件

- データファイル
  - data/SS_data/ArchiveII.csv
  - data/SS_data/ArchiveII_famfold_splits.csv
    - id 列を index として読み込みます。
    - fold 列, partition 列（"train"/"validation"/"test" など）が必要です。
- Python モジュール
  - SSpredictor パッケージ一式（SSpredictor/train.py, SSpredictor/test.py など）が存在し、同一リポジトリ内にあること。
  - 実行環境に必要なライブラリ（pandas, hydra, torch, pretrain パッケージなど）がインストールされていること。

### 必須・推奨引数

run_archiveii_famfold.py 自体に必須引数はありませんが、SSpredictor 側の設定検証により、以下のいずれかは必須です。

- 事前学習モデルを用いる場合
  - train:pretrain.framework=XXX
  - train:pretrain.timestamp=YYYYMMDDTHHMMSS
- 事前計算済み埋め込みを用いる場合
  - train:dataset.embedding_file=ファイル名.h5

上記のどちらも指定しない場合、SSpredictor/utils.py の validate_config により、train.py 実行時に次のエラーが発生します。

- ValueError("Either pretrain.framework and pretrain.timestamp or dataset.embedding_file must be specified.")

その他、よく指定する引数:

- train:model=モデル名
  - デフォルトは knotfold です。省略すると model=knotfold が自動設定されます。
- train:model.arch.use_simple=true
  - KnotFold のシンプルなアーキテクチャを使いたい場合。
- test:pretrain.framework=..., test:pretrain.timestamp=...
  - test 側だけ別の事前学習モデルを指定したい場合（通常は省略し、train と同じ設定を流用）。

### 起こりうるエラーと注意点

1. データファイルが存在しない / 列が不足している
   - data/SS_data/ArchiveII.csv, data/SS_data/ArchiveII_famfold_splits.csv が存在しない場合:
     - pandas の read_csv で FileNotFoundError が発生します。
   - splits ファイルに id, fold, partition 列が無い場合:
     - インデックス設定や splits.fold, splits.partition 参照時に KeyError が発生します。

2. family 指定が不正
   - 実行時に "family=XXX" を渡し、splits.fold に XXX が含まれていない場合:
     - run_archiveii_famfold.py 内で ValueError("Specified family 'XXX' is not in the dataset. Available families: ...") を送出して終了します。

3. 特定 family のデータが極端に少ない / 空
   - ある fold/family に対応する行が 0 件に近い場合、
     - SSpredictor/train.py 側の学習ループで total_steps が極端に小さくなる、あるいは 0 になることで、学習ロジックが想定外の挙動をする可能性があります。
   - 特に train ローダの長さが 0 の場合、CosineScheduler などでエラーになる恐れがあります。

4. 事前学習モデル・埋め込みの指定忘れ
   - 上述の通り、以下の条件を満たさないと必ず ValueError になります。
     - (pretrain.framework と pretrain.timestamp の両方が指定されている) または dataset.embedding_file が指定されている。
   - 引数忘れに注意してください。

5. test 結果の集約に関する注意
   - 各 family ごとに SSpredictor/test.py は、
     - results/SS_results/ArchiveII_famfold/[pretrain or embedding]/[model_name]/[SStrain_model_path.timestamp]/{family}/test_results/[path.timestamp]/prediction_results.csv
     に結果を書き出します。
   - run_archiveii_famfold.py の最後の集約では、これらの prediction_results.csv を読み込んで overall_results.csv を作成します。
   - 途中で学習やテストが失敗した family、手動でファイルを削除した family などは、
     - 読み込み時にファイルが見つからず Warning が出て、その family は overall_results.csv から除外されます。

### 実行時のワークフロー概要

1. run_archiveii_famfold.py 実行
   - タイムスタンプを生成し、train/test 双方に共通の path.timestamp と SStrain_model_path.timestamp を設定します。
   - experiment.name=ArchiveII_famfold を train/test 双方に設定します。
   - model 名を明示しない場合、model=knotfold を自動で設定し、test 側の SStrain_model_path.model_name に反映させます。

2. family ごとのループ
   - splits から当該 family の train/test 行を抽出し、data/SS_data/archiveii_famfold/{family}/ に train.csv, test.csv を保存します。
   - train.py を、experiment.additional_experiment_info={family} を付けて実行します。
   - test.py を、同じく experiment.additional_experiment_info={family} を付けて実行します。

3. 結果の集約
   - 各 family の prediction_results.csv を読み込み、family 列を付けて縦方向に結合し、
     - results/SS_results/ArchiveII_famfold/[...]/overall_results.csv
     として保存します。

このファイルの内容を前提に、新しいスクリプトを追加する場合や実験条件を変更する場合は、パスや引数仕様が整合しているかを確認してください。

---

## SSpredictor/train.py の設定パラメータ一覧（conf/config.py）

train.py は Hydra を通して MainConfig を受け取り、その中の各セクションで学習設定を行います。

- common: CommonConfig
  - seed (int): 乱数シード。
  - batch_size (int): 学習時のバッチサイズ。
  - max_epochs (int): 学習エポック数。
  - eval_per_epoch (int): 1 エポックあたり何回評価するか（学習ステップ数から eval 間隔を決める）。
  - eval_steps (int): 評価時に何ステップぶんのバッチを使うか。
  - use_gpu (bool): GPU を利用するかどうか。
  - validation (bool): validation split による評価を行うかどうか。
  - iterations (int): 同じ設定で学習する反復回数（重みを iteration ごとに保存）。

- pretrain: PretrainConfig
  - framework (str | None): 使用する事前学習モデルのフレームワーク名（例: "data2vec", "mlm"）。
  - timestamp (str | None): 使用する事前学習実験のタイムスタンプ（results/pretrain_results/ 以下のディレクトリ名）。
  - checkpoint (str): 読み込むチェックポイント（"final" なら max_steps、あるいは任意のステップ番号）。

- path: PathConfig
  - data_dir (str): SS データのルートディレクトリ（例: ./data/SS_data/）。
  - embedding_dir (str): 埋め込み HDF5 のディレクトリ（例: ./data/embeddings/）。
  - pretrain_model_dir (str): 事前学習モデルの保存ディレクトリ（例: ./results/pretrain_results/）。
  - output_dir (str): 二次構造予測の結果ルート（例: ./results/SS_results/${experiment.name}/）。
  - timestamp (str): 学習実行ごとのタイムスタンプ（通常は now から自動生成）。

- dataset: DatasetConfig
  - max_length (int): 入力シーケンスの最大長。これを超える部分は切り捨てる／前処理側で制御。
  - sequence_file (str): 元データ CSV のファイル名（例: ArchiveII.csv）。
  - embedding_file (str | None): 事前計算済み埋め込み HDF5 ファイル名（指定しない場合は pretrain モデルから埋め込みを生成）。
  - train_file (str): train split 用 CSV ファイル名（例: train.csv）。
  - validation_file (str): validation split 用 CSV ファイル名（例: validation.csv / test.csv）。
  - test_file (str): test split 用 CSV ファイル名（例: test.csv）。

- experiment: ExperimentConfig
  - name (str): 実験名。データパス・結果パスおよび wandb のタグに使用される必須パラメータ。
  - additional_experiment_info (str | None): 追加情報（famfold の family 名や kfold の fold 番号など）。データパスおよび出力パスのサブディレクトリに利用。
  - use_teacher (bool): 事前学習で教師モデルの出力を利用するかどうか（data2vec 用）。
  - use_attention (bool): pretrain 出力のうち attention (B, E, L, L) を使うか、表現ベクトル (B, L, E) を使うかの切り替え。

- model: ModelConfig / KnotFoldConfig
  - 共通 (ModelConfig):
    - _target_ (str): Hydra によるモデルクラスの import パス（例: SSpredictor.models.KnotFoldModel）。
    - name (str): モデル名（例: "knotfold"）。
    - lr (float): 学習率（CosineScheduler の max_lr としても使用）。
    - min_lr (float): 学習率スケジュールの下限値。
    - weight_decay (float): weight decay 係数。
  - KnotFold 特有 (KnotFoldConfig):
    - kf_lambda (float): KnotFold の min-cost-flow アルゴリズムで用いる λ。主に学習設定として保持される値。現在の test.py では `experiment.kf_lambda_cfg` の範囲走査を用いる実装のため、test 実行時に train 側の `kf_lambda` を直接フォールバック参照しない。
    - max_ref_epochs (int): 参照モデル (reference) の最大エポック数。
    - arch (KnotFoldArchConfig): KnotFold のアーキテクチャ設定。
      - use_simple (bool): 線形層 1 枚のみのシンプルなアーキテクチャを使うかどうか。
      - conv_dim (int): CNN アーキテクチャの場合のチャネル数（ResNet2D の次元）。
      - kernel_size (int): 畳み込みカーネルサイズ。
      - n_residual_blocks (int): ResNet2D の残差ブロック数。

- optimizer: OptimizeConfig / AdamWConfig
  - 共通 (OptimizeConfig):
    - _target_ (str): Hydra による optimizer クラスの import パス（例: torch.optim.AdamW）。
    - name (str): optimizer 名（例: "adamw"）。
    - lr (float): optimizer のベース学習率。
  - AdamW 特有 (AdamWConfig):
    - weight_decay (float): weight decay 係数。

---

## SSpredictor/test.py の設定パラメータ一覧（conf/test_config.py）

test.py は Hydra を通して MainConfig (テスト用) を受け取り、学習済みモデルの読み出しと評価・予測設定を行います。

- common: CommonConfig
  - seed (int): 乱数シード。
  - batch_size (int): テスト時のバッチサイズ。
  - use_gpu (bool): GPU を利用するかどうか。
  - iterations (int | None): 何回分の学習済み重みを読み込んでアンサンブルするか。`null` の場合は train_config の iterations に従う。
  - evaluation (bool): ROC/PR AUC などの評価を行うかどうか（少なくとも evaluation か prediction のどちらか一方は True である必要がある）。
  - prediction (bool): KnotFold による最終予測（knotfold/min-cost-flow）と prediction_results.csv の出力を行うかどうか。
  - save_probability_matrix (bool): 各シーケンスの予測確率行列を HDF5 と画像に保存するかどうか。

- pretrain: PretrainConfig
  - framework (str | None): 使用する事前学習モデルのフレームワーク名（通常は train_config から復元されるので省略可）。
  - timestamp (str | None): 使用する事前学習実験のタイムスタンプ（通常は train_config から復元されるので省略可）。

- path: PathConfig
  - output_dir (str): 学習済みモデルのルートディレクトリ（例: ./results/SS_results/${experiment.name}/）。
  - timestamp (str): テスト実行ごとのタイムスタンプ。test_results/<timestamp>/ 以下の出力に利用される。

- SStrain_model_path: SStrainModelPathConfig
  - model_name (str): 学習時の `cfg.model.name`（例: "knotfold"）。
  - timestamp (str): 学習時の `cfg.path.timestamp`（train_config ディレクトリや weights ディレクトリを特定するのに使用）。

- dataset: DatasetConfig
  - embedding_file (str | None): テスト時に使用する埋め込み HDF5 ファイル名（train_config 側の設定が優先されるが、ここで上書き可能）。
  - test_file (str): テスト用 CSV ファイル名（例: test.csv / family ごとの test_{fam}.csv）。train_config の data_dir, experiment.name, additional_experiment_info と組み合わせて実際のパスが決まる。

- evaluation: EvaluationConfig
  - auc_step (float): ROC AUC / PR AUC 計算の際の閾値刻み幅（例: 0.01 なら 0〜1 を 0.01 刻みで評価）。

- experiment: ExperimentConfig
  - name (str): 実験名。train_config の experiment.name と一致させる必要がある。
  - additional_experiment_info (str | None): 追加情報（famfold の family 名など）。train_config 側と一致させないと dataset/test_file のパスがずれる。
  - kf_lambda_cfg (KfLambdaConfig): KnotFold の min-cost-flow アルゴリズムで用いる λ のスキャン設定。
    - min (float | None): kf_lambda の最小値。
    - max (float | None): kf_lambda の最大値。
    - step (float | None): kf_lambda の刻み幅。
    - 現在の test.py 実装では `np.arange(min, max + step, step)` で候補列を作るため、実行時には min/max/step が数値として解決される必要がある。
    - 既定値は `SSpredictor/conf/test_config.yaml` で `min=-1.0, max=5.0, step=0.2` が設定されている。

これらのパラメータを組み合わせることで、scripts/run_archiveii_famfold.py から family ごとの学習・評価条件を柔軟に制御できます。
