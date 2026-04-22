import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    # カレントディレクトリのパスを取得
    current_dir_path = Path(__file__).parent.parent.resolve()
    
    # train.pyとtest.pyのpathを指定
    train_script_path = current_dir_path / "SSpredictor/train.py"
    test_script_path = current_dir_path / "SSpredictor/test.py"
    
    # sys.argv[0] は 'script.py' 自身だから無視して、[1:] 以降を取得する
    args = sys.argv[1:]

    train_args = []
    test_args = []
    others_args = []

    # ターミナルから渡された引数をプレフィックスを見て振り分ける
    for arg in args:
        if arg.startswith("train:"):
            # "train:" の部分を削ってリストに追加
            train_args.append(arg.replace("train:", "", 1))
        elif arg.startswith("test:"):
            # "test:" の部分を削ってリストに追加
            test_args.append(arg.replace("test:", "", 1))
        else:
            others_args.append(arg)
            print(f"Warning: Argument '{arg}' does not start with 'train:' or 'test:'. It may be ignored.")
            
    # タイムスタンプの生成
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    
    # train.pyの引数に設定を追加
    train_args.append(f"path.timestamp={timestamp}")
    train_args.append("dataset.sequence_file=ArchiveII.csv")
    train_args.append("experiment.name=ArchiveII_famfold")
    
    # test.pyの引数に設定を追加
    test_args.append(f"SStrain_model_path.timestamp={timestamp}")
    test_args.append(f"path.timestamp={timestamp}")
    test_args.append("experiment.name=ArchiveII_famfold")
    
    # train_argsにmodelの指定があるかどうかを確認する
    model_name = None
    if any(arg.startswith("model=") for arg in train_args):
        # すでにmodelの指定がある場合は、そこからmodel_nameを抽出する
        for arg in train_args:
            if arg.startswith("model="):
                model_name = arg.replace("model=", "", 1)
                test_args.append(f"SStrain_model_path.model_name={model_name}")
                break
    else:
        # modelの指定がない場合は、デフォルトでknotfoldを使用する
        model_name = "knotfold"
        train_args.append(f"model={model_name}")
        test_args.append(f"SStrain_model_path.model_name={model_name}")

    # train_argsを見てpretrainの指定があるかどうかを確認する
    # 基本はtrain_argsと同じものをtest_argsにも追加する
    pretrain_framework = None
    if any(arg.startswith("pretrain.framework=") for arg in train_args):
        for arg in train_args:
            if arg.startswith("pretrain.framework="):
                pretrain_framework = arg.replace("pretrain.framework=", "", 1)
                if pretrain_framework.startswith("[") and pretrain_framework.endswith("]"):
                    pretrain_framework = json.loads(pretrain_framework)
                else:
                    raise ValueError(f"Invalid format for pretrain.framework: {pretrain_framework}. It should be a JSON list (e.g., '[\"framework1\", \"framework2\"]') or a single string (e.g., 'framework1').")
                test_args.append(arg)
                break
            
    pretrain_timestamp = None
    if any(arg.startswith("pretrain.timestamp=") for arg in train_args):
        for arg in train_args:
            if arg.startswith("pretrain.timestamp="):
                pretrain_timestamp = arg.replace("pretrain.timestamp=", "", 1)
                if pretrain_timestamp.startswith("[") and pretrain_timestamp.endswith("]"):
                    pretrain_timestamp = json.loads(pretrain_timestamp)
                else:
                    raise ValueError(f"Invalid format for pretrain.timestamp: {pretrain_timestamp}. It should be a JSON list (e.g., '[\"timestamp1\", \"timestamp2\"]') or a single string (e.g., 'timestamp1').")
                test_args.append(arg)
                break

    # 同様にdataset.embedding_fileの指定も確認する
    # 基本はtrain_argsと同じものをtest_argsにも追加する
    dataset_embedding_file = None
    if any(arg.startswith("dataset.embedding_file=") for arg in train_args):
        for arg in train_args:
            if arg.startswith("dataset.embedding_file="):
                dataset_embedding_file = arg.replace("dataset.embedding_file=", "", 1)
                if dataset_embedding_file.startswith("[") and dataset_embedding_file.endswith("]"):
                    dataset_embedding_file = json.loads(dataset_embedding_file)
                else:
                    raise ValueError(f"Invalid format for dataset.embedding_file: {dataset_embedding_file}. It should be a JSON list (e.g., '[\"file1\", \"file2\"]') or a single string (e.g., 'file1').")
                test_args.append(arg)
                break


    # データセットの読み込みと分割
    df = pd.read_csv(current_dir_path / "data/SS_data/ArchiveII.csv", index_col="id")
    splits = pd.read_csv(current_dir_path / "data/SS_data/ArchiveII_famfold_splits.csv", index_col="id")
    
    # RNA family
    family = splits.fold.unique()
    
    ## 追加の引数で特定のfamilyを指定された場合はそのfamilyのみを使用する
    if others_args:
        for arg in others_args:
            if arg.startswith("family="):
                specified_family_list = arg.replace("family=", "", 1)
                if specified_family_list.startswith("[") and specified_family_list.endswith("]"):
                    specified_family_list = json.loads(specified_family_list)
                else:
                    raise ValueError(f"Invalid format for family: {specified_family_list}. It should be a JSON list (e.g., '[\"family1\", \"family2\"]') or a single string (e.g., 'family1').")
                
                # 指定されたfamilyがデータセットに存在するか確認する
                for specified_family in specified_family_list:
                    if not specified_family in family:
                        raise ValueError(f"Specified family '{specified_family}' is not in the dataset. Available families: {family}")
                
                family = specified_family_list
                print(f"Using specified family/families: {family}")
                break

    for fam in family:
        train = df.loc[splits[(splits.fold == fam) & (splits.partition != "test")].index]
        test = df.loc[splits[(splits.fold == fam) & (splits.partition == "test")].index]
        data_path = current_dir_path / f"data/SS_data/archiveii_famfold/{fam}/"
        os.makedirs(data_path, exist_ok=True)
        train.to_csv(data_path / "train.csv")
        test.to_csv(data_path / "test.csv")
        
        print(f"Prepared data for family '{fam}' with {len(train)} training samples and {len(test)} test samples.")
        
        # train.pyの実行
        train_args_for_fam = train_args + [f"experiment.additional_experiment_info={fam}"]
        print(f"running train.py... (additional args: {train_args_for_fam})")
        # check=Trueを指定して, train.pyがエラーを出して終了した場合にここで例外が発生するようにする
        subprocess.run(["python", str(train_script_path)] + train_args_for_fam, check=True)

        print("-" * 40)

        # test.pyの実行
        test_args_for_fam = test_args + [f"experiment.additional_experiment_info={fam}"]
        print(f"running test.py... (additional args: {test_args_for_fam})")
        subprocess.run(["python", str(test_script_path)] + test_args_for_fam, check=True)
        
    # test.pyの出力先の特定
    # f"results/SS_results/ArchiveII_famfold/{cfg.pretrain.framework}/{cfg.pretrain.timestamp}/{cfg.SStrain_model_path.model_name}/{cfg.SStrain_model_path.timestamp}/{cfg.experiment.additional_experiment_info}/test_results/{cfg.path.timestamp}/"
    test_results_dir_path = current_dir_path / f"results/SS_results/ArchiveII_famfold"
    
    ## pretrainまたはdataset.embedding_fileの情報をパスに追加
    if pretrain_framework is not None and pretrain_timestamp is not None and dataset_embedding_file is not None:
        test_results_dir_path /= "combined_representation"
    elif pretrain_framework is not None and pretrain_timestamp is not None:
        if len(pretrain_framework) == 1 and len(pretrain_timestamp) == 1:
            test_results_dir_path /= pretrain_framework[0] / pretrain_timestamp[0]
        else:
            test_results_dir_path /= "combined_representation"
    elif dataset_embedding_file is not None:
        if len(dataset_embedding_file) == 1:
            test_results_dir_path /= Path(dataset_embedding_file[0]).stem
        else:            
            test_results_dir_path /= "combined_representation"
    
    ## modelの情報をパスに追加
    if model_name is not None:
        test_results_dir_path /= model_name
        
    ## SStrain_model_path.timestampの情報をパスに追加
    test_results_dir_path /= timestamp
    
    # 各familyのtest_resultsをまとめる
    overall_results = []
    for fam in family:
        test_results_path = test_results_dir_path / fam / "test_results" / timestamp / "prediction_results.csv"
        if test_results_path.exists():
            fam_results = pd.read_csv(test_results_path)
            fam_results["family"] = fam
            overall_results.append(fam_results)
        else:
            print(f"Warning: Test results for family '{fam}' not found at {test_results_path}. Skipping.")
            
    if overall_results:
        overall_results_df = pd.concat(overall_results, ignore_index=True)
        overall_results_df.to_csv(test_results_dir_path / "overall_results.csv", index=False)
        print(f"Saved overall results to {test_results_dir_path / 'overall_results.csv'}")
        

if __name__ == "__main__":
    main()
