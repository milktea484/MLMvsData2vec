import datetime
import json
import subprocess
import sys
from pathlib import Path


def main():
    # カレントディレクトリのパスを取得
    # current_dir_path = Path(__file__).parent.parent.resolve()
    # 実行位置のパスを取得
    current_dir_path = Path.cwd().resolve()
    
    # sys.argv[0] は 'script.py' 自身だから無視して、[1:] 以降を取得する
    args = sys.argv[1:]

    pretrain_args = []

    # ターミナルから渡された引数をプレフィックスを見て振り分ける
    # 必須の引数 "framework", "timestamp", "checkpoint", "dataset.test_file"
    required_args = {
        "framework": None,
        "timestamp": None,
        "checkpoint": None,
        "test_file": None,
        "extract_repr_layers": None,
    }
    
    # 任意の引数(bool) "is_attention", "is_embedding_copy", "is_teacher"
    optional_args = {
        "is_attention": False,
        "is_embedding_copy": False,
        "is_teacher": False,
    }
    
    # 引数を振り分ける
    for arg in args:
        if any(arg.startswith(f"{req_arg}=") for req_arg in required_args):
            required_args[arg.split("=")[0]] = arg.split("=")[1]
        elif any(arg.startswith(f"{opt_arg}=") for opt_arg in optional_args):
            optional_args[arg.split("=")[0]] = arg.split("=")[1]
        else:
            print(f"Warning: Unrecognized argument '{arg}' will be ignored.")
            
    # 必須の引数がすべて揃っているか確認
    if None in required_args.values():
        missing_args = [k for k, v in required_args.items() if v is None]
        raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")
    
    # 引数を pretrain.py 用に整形
    for key, value in required_args.items():
        pretrain_args.append(f"{key}={value}")
    for key, value in optional_args.items():
        pretrain_args.append(f"{key}={value}")
    
    # pretrain.pyを実行
    pretrain_command = ["python", "-m", "pretrain.test"] + pretrain_args
    print(f"Executing pretrain.py with command: {' '.join(pretrain_command)}")
    
    try:
        subprocess.run(pretrain_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing pretrain.py:\n{e}")
        sys.exit(1)
    
    # 出力されたembeddingファイルを指定のディレクトリにコピー
    if optional_args["is_embedding_copy"] == "True":
        extract_repr_layers = json.loads(required_args["extract_repr_layers"])
        test_file_name = Path(required_args["test_file"]).stem
        embedding_type = "attn" if optional_args["is_attention"] == "True" else "repr"
        if optional_args["is_teacher"] == "True":
            embedding_type = "teacher_" + embedding_type
        
        for index, layer in enumerate(extract_repr_layers):
            embedding_file = current_dir_path / "results" / "pretrain_results" / required_args["framework"] / required_args["timestamp"] / "embeddings" / f"{test_file_name}_{embedding_type}_layer{layer}.h5"
            
            embedding_dir = current_dir_path / "data" / "embeddings"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            
            embedding_file_name = f"{required_args['framework']}_{test_file_name}_{embedding_type}_layer{layer}_embedding.h5"
            
            subprocess.run(["cp", str(embedding_file), str(embedding_dir / embedding_file_name)], check=True)
        
        ## ログ
        with embedding_dir.joinpath("embedding_log.txt").open("a") as log_file:
            log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {required_args['framework']}_{test_file_name}_{embedding_type}_layerX_embedding.h5 generated from {required_args['test_file']} using {required_args['framework']} at {required_args['timestamp']}\n")

if __name__ == "__main__":
    main()
