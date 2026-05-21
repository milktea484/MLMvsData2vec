# PrismNet 事前学習特徴統合（structure_source=pretrain）説明書

このドキュメントは、本リポジトリ内で追加した「PrismNetの入力5ch目（柔軟性/構造チャネル）を、凍結した事前学習（pretrain）モデルの特徴表現から作った1次元値で置き換える」機能について、**変更点を理解する**ことを主目的にまとめたものです。

- 対象コード範囲: `RBPpredictor/PrismNet/` 配下のみ
- 追加機能の要点:
  - `--structure_source shape|pretrain` で切替
  - `pretrain`選択時は、事前学習モデル（data2vec / MLM）を**凍結**して表現を抽出し、以下のどちらかで入力へ結合
    - `--pretrain_concat_mode proj1d`（デフォルト）: 学習可能な `Linear(embed_dim→1)` で **1次元へ射影**して PrismNet入力の5ch目として使用
    - `--pretrain_concat_mode raw`: **射影せず** `onehot(4ch)` に `repr(embed_dim)` をそのまま結合して使用（入力特徴数が `4+embed_dim` に増加）

---

## 1. 使い方（コマンド例）

以下は **PrismNet ディレクトリ（`RBPpredictor/PrismNet`）をカレント**として実行する例です。

### 1.1 pretrainを使わない（従来動作: SHAPE/柔軟性スコアをそのまま使用）

#### 学習（train）

```bash
python tools/main.py \
  --train \
  --p_name TIA1_Hela \
  --mode pu \
  --arch PrismNet \
  --data_dir data/clip_data \
  --out_dir out/my_run_shape
```

#### 評価（eval）

```bash
python tools/main.py \
  --eval --load_best \
  --p_name TIA1_Hela \
  --mode pu \
  --arch PrismNet \
  --data_dir data/clip_data \
  --out_dir out/my_run_shape
```

#### 推論（infer）

```bash
python tools/main.py \
  --infer --load_best \
  --p_name TIA1_Hela \
  --mode pu \
  --arch PrismNet \
  --out_dir out/my_run_shape \
  --infer_file /path/to/inference.tsv
```

> 重要: `shape`（デフォルト）では推論TSVに「構造列（SHAPE等）」列が必要です（後述）。

---

### 1.2 pretrainを使う（新機能: 事前学習表現 → 1次元へ削減 → 5ch目として使用）

#### 学習（train）: data2vec

```bash
python tools/main.py \
  --train \
  --structure_source pretrain \
  --pretrain_concat_mode proj1d \
  --pretrain_amp \
  --pretrain_framework data2vec \
  --pretrain_timestamp 20260324T045257 \
  --pretrain_checkpoint final \
  --p_name TIA1_Hela \
  --mode pu \
  --arch PrismNet \
  --data_dir data/clip_data \
  --out_dir out/my_run_pretrain_data2vec \
  --batch_size 32 \
  --lr 1e-4
```

#### 学習（train）: MLM

```bash
python tools/main.py \
  --train \
  --structure_source pretrain \
  --pretrain_concat_mode proj1d \
  --pretrain_amp \
  --pretrain_framework mlm \
  --pretrain_timestamp 20260423T083459 \
  --pretrain_checkpoint final \
  --p_name TIA1_Hela \
  --mode pu \
  --arch PrismNet \
  --data_dir data/clip_data \
  --out_dir out/my_run_pretrain_mlm \
  --batch_size 32 \
  --lr 1e-4
```

#### 評価（eval）

```bash
python tools/main.py \
  --eval --load_best \
  --structure_source pretrain \
  --pretrain_concat_mode proj1d \
  --pretrain_amp \
  --pretrain_framework data2vec \
  --pretrain_timestamp 20260324T045257 \
  --pretrain_checkpoint final \
  --p_name TIA1_Hela \
  --mode pu \
  --arch PrismNet \
  --data_dir data/clip_data \
  --out_dir out/my_run_pretrain_data2vec \
  --batch_size 32
```


補足: 推論データが `.h5` の場合は `--infer_file /path/to/some.h5` として渡すと、そのh5内のtest splitに対して推論します。
また、学習に使った `p_name` のh5のtest splitに推論したいだけなら `--infer_test` が使えます。
#### 推論（infer）

```bash
python tools/main.py \
  --infer --load_best \
  --structure_source pretrain \
  --pretrain_concat_mode proj1d \
  --pretrain_amp \
  --pretrain_framework data2vec \
  --pretrain_timestamp 20260324T045257 \
  --pretrain_checkpoint final \
  --p_name TIA1_Hela \
  --mode pu \
  --arch PrismNet \
  --out_dir out/my_run_pretrain_data2vec \
  --infer_file /path/to/inference.tsv \
  --batch_size 32
```

> 重要: `--structure_source pretrain` のとき、推論TSVは **構造列が不要**です（内部で `use_structure=False` にして読み込みます）。

### 1.3 `*_Hela.h5` を逐次実行（複数タンパクをまとめて回す）

`data/clip_data` 配下の `*_Hela.h5` を検出して、**1つずつ順に** `tools/main.py` を実行するためのシェルが `tools/run_all_hela_h5.sh` です。

- 出力は `out/batch_<timestamp>/<protein>/` のように **実行時刻（timestamp）を自動採番**し、さらにタンパクごとに分けるため、基本的に上書きが発生しません。
- `tools/run_all_hela_h5.sh` の後ろに渡した引数は、そのまま `tools/main.py` に転送されます。

注意（`--load_best` について）:
- `tools/run_all_hela_h5.sh` は内部で常に `--train` を付けて実行します。`tools/main.py` は学習終了後に `*_best.pth` を**自動で読み直す**ため、同一プロセス内で続けて `--eval` / `--infer_test` を行う場合は `--load_best` を明示しなくても best で評価・推論されます。
- 一方で、学習とは別プロセスで **`--eval` や `--infer_test` だけ**を実行する場合は、学習済み重みを読むために `--load_best` が必要です（付けないと初期重みのまま評価・推論します）。

#### 逐次実行（shape; 学習+評価）

```bash
bash tools/run_all_hela_h5.sh \
  --eval \
  --infer_test \
  --batch_size 64
```

#### 逐次実行（pretrain; 学習+評価+test推論）

```bash
bash tools/run_all_hela_h5.sh \
  --eval \
  --infer_test \
  --structure_source pretrain \
  --pretrain_concat_mode proj1d \
  --pretrain_framework data2vec \
  --pretrain_timestamp 20260324T045257 \
  --pretrain_checkpoint final \
  --pretrain_amp \
  --batch_size 32

#### 逐次実行（pretrain; raw concatモードの例）

```bash
bash tools/run_all_hela_h5.sh \
  --eval \
  --infer_test \
  --structure_source pretrain \
  --pretrain_concat_mode raw \
  --pretrain_framework data2vec \
  --pretrain_timestamp 20260324T045257 \
  --pretrain_checkpoint final \
  --pretrain_amp \
  --batch_size 32
```
```

#### 逐次実行（pretrain; teacher重みを使う例）

```bash
bash tools/run_all_hela_h5.sh \
  --eval \
  --infer_test \
  --structure_source pretrain \
  --pretrain_framework mlm \
  --pretrain_timestamp 20260316T030756 \
  --pretrain_checkpoint final \
  --pretrain_amp \
  --batch_size 32
```

---

## 2. 追加したCLIオプション（tools/main.py）

追加した引数は以下です（`tools/main.py` で定義）。

- `--structure_source {shape,pretrain}`
  - `shape`（デフォルト）: 従来通り、入力データに含まれる構造/柔軟性スコア（5ch目）を使用
  - `pretrain`: **凍結pretrainモデルの表現から生成した1次元**で5ch目を置き換え

- `--pretrain_framework {data2vec,mlm}`
  - 事前学習側のフレームワーク選択

- `--pretrain_timestamp <TIMESTAMP>`（必須; `structure_source=pretrain` のとき）
  - `results/pretrain_results/<framework>/<timestamp>/` の `<timestamp>` 部分

- `--pretrain_model_root <DIR>`
  - 省略時は、`tools/main.py` の位置から推定して `<workspace>/results/pretrain_results` がデフォルトになります

- `--pretrain_checkpoint <N|final>`
  - `final` の場合、pretrain設定の `common.max_steps` を使用して `weight_<max_steps>.pth` を探します

- `--pretrain_use_teacher`
  - `weight_<...>.pth` の代わりに `teacher_weight_<...>.pth` を使用

- `--pretrain_amp`
  - `structure_source=pretrain` のとき、凍結pretrain forward を AMP（autocast fp16/bf16）で実行して高速化します

- `--pretrain_concat_mode {proj1d,raw}`
  - `proj1d`（デフォルト）: `Linear(embed_dim→1)` で1次元化して 5ch目として使用（従来の追加機能）
  - `raw`: **1次元へ圧縮せず** `repr(embed_dim)` を `onehot(4ch)` にそのまま結合して使用

- `--infer_test`
  - `--infer_file`（TSV）を使わず、学習用h5内のtest splitに対して推論します（`*_Hela.h5` を推論データとして扱う用途）

また、推論時の入力形式差を吸収するために、`--structure_source pretrain` の場合は
`SeqicSHAPE(..., use_structure=False)` となるように変更しています。

---

## 3. 追加した処理の詳細（理解のための内部説明）

ここが本機能の中核です。実装は主に `prismnet/model/PrismNet.py` に入っています。

### 3.1 何を置き換えるのか（PrismNet入力の5ch目）

PrismNet（`mode=pu`）の入力は概ね「(A,C,G,Uのone-hot 4ch) + (構造/柔軟性 1ch)」の **計5ch** です。

新機能では、`--structure_source pretrain` のとき **5ch目を以下で置き換え**ます:

1. 入力の one-hot 4ch を取り出す
2. one-hot を pretrain用のトークン列に変換
3. 凍結した pretrainモデルで各位置の表現（`repr`）を抽出
4. その表現を学習可能な線形層で `embed_dim → 1` に射影
5. (B,1,L,1) に整形して one-hot 4ch と結合し、(B,1,L,5) を作る

### 3.2 pretrainモデルのロード手順（Hydra/OmegaConf）

`structure_source=pretrain` を指定すると、PrismNetの初期化時に以下を行います。

- pretrain実験ディレクトリを組み立て
  - `pretrain_model_dir = <pretrain_model_root>/<framework>/<timestamp>`
- pretrainのHydra config を読む
  - `train_config/.hydra/config.yaml`
- Hydraで pretrain framework を instantiate
  - ただし古い実験で `_target_` が `models.data2vecModel` のように短い場合があるため、
    `pretrain.models.data2vecModel` / `pretrain.models.MLMModel` に補正しています
- OmegaConf の custom resolver を登録
  - pretrain側configに `${div:...}` `${mul:...}` が含まれるため、利用側（PrismNet）でも resolver 登録が必要です
- checkpointの重みファイルをロード
  - `weight_<ckpt>.pth` または `teacher_weight_<ckpt>.pth`
  - `ckpt=final` の場合は `common.max_steps` を使用

### 3.3 凍結の方針（「pretrainは固定、PrismNetは学習」）

本機能は「pretrainモデルを凍結して特徴抽出器として使う」設計です。

- pretrainモデル:
  - `eval()` 固定
  - 全パラメータ `requires_grad=False`
  - forward（`_test`）は `torch.no_grad()` で実行
- 学習対象:
  - PrismNet本体のパラメータ
  - `pretrain_proj = Linear(embed_dim, 1)`（**この1層だけ学習**）

補足:
- `torch.inference_mode()` を使うと、`Linear` に渡したテンソルが autograd 的に制限されて
  backwardでエラーになるケースがあったため、`no_grad()` を採用しています。

### 3.4 one-hot → pretrainトークン化の仕様

PrismNet入力（B,1,L,4）の one-hot を、pretrainの `dataset.tokens` に基づいてID列へ変換します。

- `argmax` で A/C/G/U を 0..3 に復元
- ただし one-hot が全ゼロ（unknown）な位置は `N` に置換
- `experiment.use_additional_token=true` の場合は `(<cls> ... <eos>)` を前後に付与

この挙動により、推論時に構造列を省略しても（=4ch入力でも）pretrain表現から5ch目を生成できます。

### 3.5 pretrain表現（repr）の取り出しと形状合わせ

pretrain側は `pretrain_model._test({...})` を呼び、返ってきた辞書から `repr` を取得します（`--pretrain_repr_key` は現状固定で `repr` を渡しています）。

- additional token あり: `repr` の長さが `L+2` になるので、先頭と末尾（cls/eos）を落として `L` に合わせます
- 想定外の長さの場合はエラーにします

得られた `repr: (B, L, embed_dim)` を `Linear(embed_dim→1)` して `(B, L, 1)`、さらに `(B, 1, L, 1)` に変形して結合します。

---

## 4. 実行に必要な環境（ライブラリ/依存関係）

### 4.1 PrismNet側（従来から必要）

- `torch`（本リポジトリ環境では PyTorch 2.5.1 + CUDA 12.1 を想定）
- `numpy`, `h5py`, `scikit-learn`, `tqdm`, `matplotlib`, `pandas`
- `termcolor`, `tensorboardX`, `einops`

※ `requirements.txt` は upstream の古いバージョン指定（`torch==1.1.0` 等）を含むため、
既にconda環境で新しいPyTorchを入れている場合は、安易にそのまま `pip install -r requirements.txt` すると依存衝突する可能性があります。

### 4.2 pretrain統合を使う場合に追加で必要

- `hydra-core`
- `omegaconf`
- `pretrain` パッケージ（このリポジトリ内の `pretrain/`）

`pretrain` を import できる必要があります（`import pretrain.models`）。
確実な方法は editable install です:

```bash
# ワークスペースroot（mystudy）で
pip install -e pretrain
```

また、pretrainの実験結果（configと重み）が以下の構造で存在する必要があります:

```
results/pretrain_results/
  <data2vec|mlm>/
    <timestamp>/
      train_config/.hydra/config.yaml
      weight_<ckpt>.pth
      (optional) teacher_weight_<ckpt>.pth
```

---

## 5. 入力データ形式（特にinfer TSV）

### 5.1 `--structure_source shape` の推論TSV

`prismnet/utils/datautils.py:load_testset_txt` は、推論TSVの列を次のように参照します:

- `line[2]`: 101ntの配列（sequence）
- `line[3]`: 構造/柔軟性スコア列（カンマ区切り）※ `use_structure=True` の場合

つまり `shape` のときは **構造列が必須**です。

### 5.2 `--structure_source pretrain` の推論TSV

`tools/main.py` 側で `use_structure=False` に切り替えるため、推論TSVは
- `line[2]` の sequence だけあれば動作します（構造列不要）

---

## 6. 実務上の注意点（ハマりどころ）

### 6.1 推論データのキャッシュ（*_test.npz）に注意

`load_testset_txt` は、初回読み込み時に `(<infer_file> + "_test.npz")` を作り、次回以降は
**use_structureの指定に関係なく** それをロードします。

そのため、同じ `infer_file` について
- 以前 `shape` で実行して `_test.npz` が作られている
- 今回 `pretrain`（`use_structure=False`）で実行したい

といった場合、古いキャッシュが残っていると入力チャネル数が意図とズレる可能性があります。

対策:
- `--structure_source` を変えて同じ `infer_file` を使う場合は、対応する `*_test.npz` を削除してください。

### 6.2 出力は同名なら上書きされます

現状の保存名は主に `identity = p_name + '_' + arch + '_' + mode` で固定で、
同じ `--out_dir` で繰り返すと `out/models/<identity>_best.pth` や `out/evals/<identity>.*` が上書きされます。

運用上の対策:
- 実行ごとに `--out_dir` を変える（例: `--out_dir out/20260513T120000_pretrain_data2vec`）

### 6.3 `out/` 配下に出るファイルの意味（metrics / probs）

PrismNetは `--out_dir <DIR>` を指定すると、基本的に `<DIR>/out/` 配下へ結果を書き出します。

- `out/evals/<identity>.metrics`
  - 1行のTSVで、**評価セット（通常はh5内のtest split）**の要約指標です。
  - 列は順に: `dataset_name` / `acc` / `auc` / `prc` / `tp` / `tn` / `fp` / `fn`
- `out/evals/<identity>.probs`
  - 評価セット各サンプルの **予測確率と正解ラベル**。
  - 1行が `prob\tlabel`（probはsigmoid後の確率、labelは0/1）。
- `out/infer/<identity>*.probs`
  - 推論（`--infer` / `--infer_test`）の出力で、各サンプルの **予測確率のみ**（1行1つ）。
- `out/models/<identity>_best.pth` / `<identity>_<epoch>.pth`
  - 学習中に保存されるモデル重み（`.pth`）。

`identity` は概ね `<p_nameのstem>_<arch>_<mode>`（例: `TIA1_Hela_PrismNet_pu`）です。

### 6.4 どの重みを読んでいるか（teacher/final）

- `--pretrain_checkpoint final` は、pretrain config の `common.max_steps` を参照して
  `weight_<max_steps>.pth` を探します
- `--pretrain_use_teacher` を付けると `teacher_weight_<...>.pth` を探します

---

## 7. 変更箇所（どこを見れば理解できるか）

- CLI引数追加と pretrain/shape の推論入力切替
  - `tools/main.py`
- pretrainロード、凍結、repr→1D射影、入力5ch目置換
  - `prismnet/model/PrismNet.py`（`PrismNet` と `PrismNet_large` の両方）
- 推論TSV読み込みの `use_structure` 対応とキャッシュ挙動
  - `prismnet/utils/datautils.py`（`load_testset_txt`）

- `*_Hela.h5` を逐次実行（上書き回避のtimestamp付きout_dir）
  - `tools/run_all_hela_h5.sh`

参考: 既存の一括学習スクリプトとして `exp/prismnet/train_all.sh`（`data/clip_data/all.list` を読む）もありますが、出力先が固定のため再実行時の上書き回避には注意が必要です。

---

## 8. FAQ

### Q1. pretrainを使うとき、訓練データ（h5）内の構造チャネルはどうなりますか？
A. 入力自体には構造チャネルが入っていても、`forward` 内で one-hot4ch から pretrain由来の1chを計算し、結合し直すので、元の構造チャネルは実質的に使われません。

### Q2. pretrainモデルも学習されますか？
A. いいえ。pretrainモデルは `eval()` 固定・`requires_grad=False`・`no_grad()` で呼ぶ設計です。学習されるのは `pretrain_proj(Linear)` と PrismNet本体です。
