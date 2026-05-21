import argparse
import os
import random
import sys
from pathlib import Path

# Allow running this file directly without installing PrismNet as a package.
# When executed as `python /.../tools/main.py`, sys.path[0] becomes `tools/`,
# so we add the project root (which contains `prismnet/`).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import prismnet.model as arch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prismnet import (
    compute_high_attention_region,
    compute_saliency,
    compute_saliency_img,
    inference,
    log_print,
    train,
    validate,
)
from prismnet.loader import SeqicSHAPE

#compute_high_attention_region
# from prismnet.engine.train_loop import 
from prismnet.model.utils import GradualWarmupScheduler
from prismnet.utils import datautils
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler


def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # print("[Info] cudnn.deterministic set to True. CUDNN-optimized code may be slow.")

def save_evals(out_dir, filename, dataname, predictions, label, met):
    evals_dir = datautils.make_directory(out_dir, "out/evals")
    metrics_path = os.path.join(evals_dir, filename+'.metrics')
    probs_path = os.path.join(evals_dir, filename+'.probs')
    with open(metrics_path,"w") as f:
        if "_reg" in filename:
            print("{:s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:d}\t{:d}\t{:d}\t{:d}\t{:.3f}\t{:.3f}\t{:.3f}".format(
                dataname,
                met.acc,
                met.auc,
                met.prc,
                met.tp,
                met.tn,
                met.fp,
                met.fn,
                met.avg[7],
                met.avg[8],
                met.avg[9],
            ), file=f)
        else:
            print("{:s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:d}\t{:d}\t{:d}\t{:d}".format(
                dataname,
                met.acc,
                met.auc,
                met.prc,
                met.tp,
                met.tn,
                met.fp,
                met.fn,
            ), file=f)
    with open(probs_path,"w") as f:
        for i in range(len(predictions)):
            print("{:.3f}\t{}".format(predictions[i,0], label[i,0]), file=f)
    print("Evaluation file:", metrics_path)
    print("Prediction file:", probs_path)

def save_infers(out_dir, filename, predictions):
    evals_dir = datautils.make_directory(out_dir, "out/infer")
    probs_path = os.path.join(evals_dir, filename+'.probs')
    with open(probs_path,"w") as f:
        for i in range(len(predictions)):
            print("{:f}".format(predictions[i,0]), file=f)
    print("Prediction file:", probs_path)

def main():
    global writer, best_epoch
    # Training settings
    parser = argparse.ArgumentParser(description='Official version of PrismNet')
    # Data options
    parser.add_argument('--data_dir',       type=str, default="data", help='data path')
    parser.add_argument('--exp_name',       type=str, default="cnn", metavar='N', help='experiment name')
    parser.add_argument('--p_name',         type=str, default="TIA1_Hela", metavar='N', help='protein name')
    parser.add_argument('--out_dir',        type=str, default=".", help='output directory')
    parser.add_argument('--mode',           type=str, default="pu", help='data mode')
    parser.add_argument("--infer_file",     type=str, help="infer file", default="")
    # Pretrain feature options (replace SHAPE/flexibility channel)
    parser.add_argument(
        "--structure_source",
        type=str,
        default="shape",
        choices=["shape", "pretrain"],
        help="Structure/flexibility channel source: shape (default, from input) or pretrain (frozen pretrain repr -> linear -> 1D).",
    )
    parser.add_argument(
        "--pretrain_framework",
        type=str,
        default="data2vec",
        choices=["data2vec", "mlm"],
        help="Which pretrain framework to load when structure_source=pretrain.",
    )
    parser.add_argument(
        "--pretrain_timestamp",
        type=str,
        default="",
        help="Pretrain run timestamp directory name under pretrain_model_root (e.g. 20260324T045257).",
    )
    parser.add_argument(
        "--pretrain_model_root",
        type=str,
        default="",
        help="Root dir that contains pretrain_results/<framework>/<timestamp> (default: <workspace>/results/pretrain_results).",
    )
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default="final",
        help="Checkpoint step number (e.g. 90000) or 'final' (uses pretrain cfg common.max_steps).",
    )
    parser.add_argument(
        "--pretrain_use_teacher",
        action='store_true',
        help="Use teacher weights (teacher_weight_*.pth) instead of weight_*.pth.",
    )
    parser.add_argument(
        "--pretrain_amp",
        action='store_true',
        help="Enable AMP (autocast fp16/bf16) for frozen pretrain forward to speed up structure_source=pretrain.",
    )
    parser.add_argument(
        "--pretrain_concat_mode",
        type=str,
        default="proj1d",
        choices=["proj1d", "raw"],
        help="How to merge pretrain features when structure_source=pretrain: proj1d (default; Linear(embed_dim->1)) or raw (concat full repr without compression).",
    )
    # Training Hyper-parameter
    parser.add_argument('--arch',           default="PrismNet", help='network architecture')
    parser.add_argument('--lr_scheduler',   default="warmup", help=' lr scheduler: warmup/cosine')
    parser.add_argument('--lr',             type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size',     type=int, default=64, help='input batch size')
    parser.add_argument('--nepochs',        type=int, default=200, help='number of epochs to train')
    parser.add_argument('--pos_weight',     type=int, default=2, help='positive class weight')
    parser.add_argument('--weight_decay',   type=float, default=1e-6, help='weight decay, default=1e-6')
    parser.add_argument('--early_stopping', type=int, default=20, help='early stopping')
    # Training 
    parser.add_argument('--load_best',      action='store_true', help='load best model')
    parser.add_argument('--eval',           action='store_true', help='eval mode')
    parser.add_argument('--train',          action='store_true', help='train mode')
    parser.add_argument('--infer',          action='store_true', help='infer mode')
    parser.add_argument('--infer_test',     action='store_true', help='infer test from h5')
    parser.add_argument('--eval_test',      action='store_true', help='eval test from h5')
    parser.add_argument('--saliency',       action='store_true', help='compute saliency mode')
    parser.add_argument('--saliency_img',   action='store_true', help='compute saliency and plot image mode')
    parser.add_argument('--har',            action='store_true', help='compute highest attention region')
    # misc
    parser.add_argument('--tfboard',        action='store_true', help='tf board')
    parser.add_argument('--no-cuda',        action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--workers',        type=int, help='number of data loading workers', default=2)
    parser.add_argument('--log_interval',   type=int, default=100, help='log print interval')
    parser.add_argument('--seed',           type=int, default=1024, help='manual seed')
    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.mode == 'pu':
        args.nstr = 1
    else:
        args.nstr = 0

    # Resolve dataset path and identity.
    # - Historical usage: --data_dir data/clip_data --p_name TIA1_Hela -> data/clip_data/TIA1_Hela.h5
    # - Newer convenience: allow --p_name to be a .h5 filename or a path.
    p_name_path = Path(args.p_name)
    p_stem = p_name_path.stem if p_name_path.suffix == ".h5" else p_name_path.name
    if p_name_path.suffix == ".h5":
        if p_name_path.is_absolute() or str(p_name_path.parent) != ".":
            # args.p_name is already a (relative/absolute) path like data/clip_data/TIA1_Hela.h5
            data_path = str(p_name_path)
        else:
            # args.p_name is a filename like TIA1_Hela.h5
            data_path = str(Path(args.data_dir) / p_name_path)
    else:
        data_path = str(Path(args.data_dir) / f"{args.p_name}.h5")

    identity   = p_stem+'_'+args.arch+"_"+args.mode
    datautils.make_directory(args.out_dir,"out/")
    model_dir  = datautils.make_directory(args.out_dir,"out/models")
    model_path = os.path.join(model_dir, identity+"_{}.pth")

    if args.tfboard:
        tfb_dir  = datautils.make_directory(args.out_dir,"out/tfb")
        writer = SummaryWriter(tfb_dir)
    else:
        writer = None
    # fix random seed
    fix_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    
    #train_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path), \
    #    batch_size=args.batch_size, shuffle=True,  **kwargs)
    #test_loader  = torch.utils.data.DataLoader(SeqicSHAPE(data_path, is_test=True), \
    #    batch_size=args.batch_size*8, shuffle=False, **kwargs)
    #print("Train set:", len(train_loader.dataset))
    #print("Test  set:", len(test_loader.dataset))


    # Default pretrain root: resolved relative to workspace (based on this file location)
    if args.pretrain_model_root == "":
        workspace_root = Path(__file__).resolve().parents[3]  # .../mystudy
        args.pretrain_model_root = str(workspace_root / "results" / "pretrain_results")

    if args.structure_source == "pretrain" and args.pretrain_timestamp == "":
        raise ValueError("--pretrain_timestamp is required when --structure_source=pretrain")

    print("Network Arch:", args.arch)
    model = getattr(arch, args.arch)(
        mode=args.mode,
        structure_source=args.structure_source,
        pretrain_concat_mode=args.pretrain_concat_mode,
        pretrain_model_root=args.pretrain_model_root,
        pretrain_framework=args.pretrain_framework,
        pretrain_timestamp=args.pretrain_timestamp if args.pretrain_timestamp != "" else None,
        pretrain_checkpoint=args.pretrain_checkpoint,
        pretrain_use_teacher=args.pretrain_use_teacher,
        pretrain_repr_key="repr",
        pretrain_amp=args.pretrain_amp,
        device=device,
    )
    arch.param_num(model)
    # print(model)

    if args.load_best:
        filename = model_path.format("best")
        print("Loading model: {}".format(filename))
        model.load_state_dict(torch.load(filename,map_location='cpu'))
 
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))

    if args.train:

        train_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path), \
            batch_size=args.batch_size, shuffle=True,  **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(SeqicSHAPE(data_path, is_test=True), \
            batch_size=args.batch_size*8, shuffle=False, **kwargs)
        print("Train set:", len(train_loader.dataset))
        print("Test  set:", len(test_loader.dataset))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=8, total_epoch=float(args.nepochs), after_scheduler=None)

        best_auc = 0
        best_acc = 0
        best_epoch = 0
        for epoch in range(1, args.nepochs + 1):
            t_met       = train(args, model, device, train_loader, criterion, optimizer)
            v_met, _, _ = validate(args, model, device, test_loader, criterion)
            scheduler.step(epoch)
            lr = scheduler.get_lr()[0]
            color_best='green'
            if best_auc < v_met.auc:
                best_auc = v_met.auc
                best_acc = v_met.acc
                best_epoch = epoch
                color_best = 'red'
                filename = model_path.format("best")
                torch.save(model.state_dict(), filename)
            if epoch - best_epoch > args.early_stopping:
                print("Early stop at %d, %s "%(epoch, args.exp_name))
                break

            if args.tfboard and writer is not None:
                writer.add_scalar('loss/train', t_met.other[0], epoch)
                writer.add_scalar('acc/train', t_met.acc, epoch)
                writer.add_scalar('AUC/train', t_met.auc, epoch)
                writer.add_scalar('lr', lr, epoch)
                writer.add_scalar('loss/test', v_met.other[0], epoch)
                writer.add_scalar('acc/test', v_met.acc, epoch)
                writer.add_scalar('AUC/test', v_met.auc, epoch)
            line='{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} lr: {:.6f}'.format(\
                args.p_name, epoch, t_met.other[0], t_met.acc, t_met.auc, lr)
            log_print(line, color='green', attrs=['bold'])
            
            line='{} \t Test  Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} ({:.4f})'.format(\
                args.p_name, epoch, v_met.other[0], v_met.acc, v_met.auc, best_auc)
            log_print(line, color=color_best, attrs=['bold'])
            
        print("{} auc: {:.4f} acc: {:.4f}".format(args.p_name, best_auc, best_acc))

        filename = model_path.format("best")
        print("Loading model: {}".format(filename))
        model.load_state_dict(torch.load(filename))

    
    
    if args.eval:

        test_loader  = torch.utils.data.DataLoader(SeqicSHAPE(data_path, is_test=True), \
            batch_size=args.batch_size*8, shuffle=False, **kwargs)
        print("Test  set:", len(test_loader.dataset))

        met, y_all, p_all = validate(args, model, device, test_loader, criterion)
        print("> eval {} auc: {:.4f} acc: {:.4f}".format(args.p_name, met.auc, met.acc))
        save_evals(args.out_dir, identity, args.p_name, p_all, y_all, met)

    if args.infer_test:
        # Inference on the test split inside the training h5 (no TSV required)
        test_loader  = torch.utils.data.DataLoader(
            SeqicSHAPE(data_path, is_test=True),
            batch_size=args.batch_size*8,
            shuffle=False,
            **kwargs,
        )
        p_all = inference(args, model, device, test_loader)
        save_infers(args.out_dir, identity + "_test", p_all)

    if args.infer and os.path.exists(args.infer_file):
        if args.infer_file.endswith(".h5"):
            test_loader = torch.utils.data.DataLoader(
                SeqicSHAPE(args.infer_file, is_test=True),
                batch_size=args.batch_size * 8,
                shuffle=False,
                **kwargs,
            )
            p_all = inference(args, model, device, test_loader)
            infer_tag = Path(args.infer_file).stem
            save_infers(args.out_dir, identity + "_" + infer_tag, p_all)
        else:
            use_structure = False if args.structure_source == "pretrain" else True
            test_loader  = torch.utils.data.DataLoader(
                SeqicSHAPE(args.infer_file, is_infer=True, use_structure=use_structure),
                batch_size=args.batch_size,
                shuffle=False,
                **kwargs,
            )
            p_all = inference(args, model, device, test_loader)
            identity = identity+"_"+ os.path.basename(args.infer_file).replace(".txt","") 
            save_infers(args.out_dir, identity, p_all)

    if args.saliency and os.path.exists(args.infer_file):
        use_structure = False if args.structure_source == "pretrain" else True
        test_loader  = torch.utils.data.DataLoader(SeqicSHAPE(args.infer_file, is_infer=True, use_structure=use_structure), \
            batch_size=args.batch_size, shuffle=False, **kwargs)
        compute_saliency(args, model, device, test_loader, identity)

    if args.saliency_img and os.path.exists(args.infer_file):
        use_structure = False if args.structure_source == "pretrain" else True
        test_loader  = torch.utils.data.DataLoader(SeqicSHAPE(args.infer_file, is_infer=True, use_structure=use_structure), \
            batch_size=args.batch_size, shuffle=False, **kwargs)
        compute_saliency_img(args, model, device, test_loader, identity)
    
    if args.har and os.path.exists(args.infer_file):
        use_structure = False if args.structure_source == "pretrain" else True
        test_loader  = torch.utils.data.DataLoader(SeqicSHAPE(args.infer_file, is_infer=True, use_structure=use_structure), \
            batch_size=args.batch_size, shuffle=False, **kwargs)
        compute_high_attention_region(args, model, device, test_loader, identity)


    
    

if __name__ == '__main__':
    main()
