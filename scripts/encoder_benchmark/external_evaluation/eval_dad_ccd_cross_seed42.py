#!/usr/bin/env python3
"""Inference-only cross-dataset evaluation, seed 42.

DAD -> CCD: V-JEPA 2.1-L, VideoMAE V1-L, VideoMAE V2-S, InternVideo2-S.
CCD -> DAD: VideoMAE V2-S.

Requires train_dad_frontier_fast_ram.py and train_ccd_videomaev2_s_ram.py
in the same directory (previously used for training). Does not retrain or
select checkpoints on the target set. Reuses compatible test frame caches.
"""
import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, f1_score, average_precision_score,
                             roc_auc_score, confusion_matrix)

DAD_MODELS = ('vjepa2_1_l', 'videomae_v1_l', 'videomaev2_s', 'internvideo2_s')


def import_script(filepath, tag):
    if not filepath.is_file():
        raise FileNotFoundError(f'Required original training script missing: {filepath}')
    spec = importlib.util.spec_from_file_location(tag, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tag] = mod
    spec.loader.exec_module(mod)
    return mod


def read_ccd_test(ccd):
    manifest = ccd / 'splits' / 'test.txt'
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    items=[]
    seen=set()
    for lineno, ln in enumerate(manifest.read_text().splitlines(), 1):
        if not ln.strip():
            continue
        parts=ln.split()
        if len(parts)!=2:
            raise ValueError(f'{manifest}:{lineno}: expected path and label')
        rel, label=parts
        p=Path(rel)
        if len(p.parts)!=2 or p.is_absolute() or '..' in p.parts or p.suffix.lower()!='.mp4':
            raise ValueError(f'Bad CCD path: {rel}')
        y=int(label)
        if (p.parts[0],y) not in (('positive',1),('negative',0)):
            raise ValueError(f'CCD folder/label mismatch: {rel} {label}')
        if rel in seen:
            raise ValueError(f'Duplicate CCD file: {rel}')
        seen.add(rel)
        file=ccd/('Crash-1500' if y else 'Normal')/p.name
        if not file.is_file():
            raise FileNotFoundError(file)
        items.append((str(file.resolve()),y,rel))
    if len(items)!=900 or sum(y for _,y,_ in items)!=300:
        raise ValueError(f'CCD official test expected 900 videos, 300 positives; got {len(items)} / {sum(y for _,y,_ in items)}')
    return items


def read_dad_test(dad, dadmodule):
    items=dadmodule.read_dad(dad,'test')
    if len(items)!=466 or sum(y for _,y in items)!=165:
        raise ValueError(f'DAD official test expected 466 videos, 165 positives; got {len(items)} / {sum(y for _,y in items)}')
    return [(path,y,str(Path(path).relative_to(dad/'videos'))) for path,y in items]


class ArrayDataset(Dataset):
    def __init__(self, array, entries):
        if len(array)!=len(entries):
            raise ValueError('Frame cache and labels have different lengths')
        self.array=array
        self.entries=entries
    def __len__(self):
        return len(self.entries)
    def __getitem__(self,i):
        # Convert readonly mmap to a writable per-sample tensor only when needed.
        frame = np.asarray(self.array[i])
        if not frame.flags.writeable:
            frame = np.array(frame, copy=True)
        return torch.from_numpy(frame).permute(0,3,1,2), int(self.entries[i][1])


def load_target_cache(dadmod, ccdmod, entries, target, cfg, args):
    dims=(cfg['frames'],cfg['width'],cfg['height'])
    # CCD 16x224: reuse exact CCD V1-S/V2-S cache, including saved sampling order.
    if target=='CCD' and dims==(16,224,224):
        rows, manifest=ccdmod.ccd_entries(args.ccd,'test')
        if [r[0] for r in rows]!=[r[2] for r in entries]:
            raise ValueError('CCD cache row order mismatch')
        path=ccdmod.ccd_cache(rows,manifest,args.ccd,'test',False)
        arr=np.load(path,mmap_mode='r')
        print('Reusing CCD original 16x224 cache:',path,flush=True)
        return ArrayDataset(arr,entries)
    # DAD model-specific cache loader also works for CCD absolute paths. It
    # validates an input-file hash and reuses any matching previously decoded file.
    cache= args.dad / 'cache' / 'frontier_ram' if target=='DAD' else args.ccd / 'cache' / 'cross_eval'
    subdir=cache / f'{cfg["frames"]}f_{cfg["width"]}x{cfg["height"]}'
    raw=[(p,y) for p,y,_ in entries]
    ds=dadmod.DADRamVideos(raw,cfg,subdir,'test')
    return ds


def eval_once(mod, model, cfg, loader, entries):
    model.eval()
    y_all=[];s_all=[]
    amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for x,y in loader:
            with torch.autocast('cuda',dtype=amp_dtype):
                logits,_=mod.forward_batch(model,x,y,cfg,False)
            s_all.extend(logits.float().softmax(1)[:,1].cpu().tolist())
            y_all.extend(y.tolist())
    if len(y_all)!=len(entries):
        raise ValueError('Number of predictions does not match target manifest')
    if y_all!=[r[1] for r in entries]:
        raise ValueError('Evaluation label ordering mismatch')
    pred=[int(s>=0.5) for s in s_all]
    metrics={
        'accuracy':float(accuracy_score(y_all,pred)),
        'macro_f1':float(f1_score(y_all,pred,average='macro',zero_division=0)),
        'accident_f1':float(f1_score(y_all,pred,zero_division=0)),
        'video_ap':float(average_precision_score(y_all,s_all)),
        'roc_auc':float(roc_auc_score(y_all,s_all)),
        'confusion_matrix':confusion_matrix(y_all,pred,labels=[0,1]).tolist(),
        'n_test':len(entries),'n_positive':sum(y_all),
        'threshold':0.5,
    }
    return metrics,y_all,s_all,pred


def run_one(name,origin,target,mod,args,entries):
    out=args.out / f'{origin.lower()}_to_{target.lower()}' / name / 'seed_42'
    out.mkdir(parents=True,exist_ok=True)
    ckpt=(args.dad / 'runs' / 'frontier_models' / name / 'seed_42' / 'best.pt'
          if origin=='DAD' else args.ccd / 'runs' / 'videomaev2_s' / 'seed_42' / 'best.pt')
    if not ckpt.is_file():
        raise FileNotFoundError(f'Saved fine-tuned checkpoint not found: {ckpt}')
    cfg=dict(mod.SPECS[name])
    print(f'\n=== {origin} -> {target}: {name}, seed 42 ===',flush=True)
    print('Checkpoint:',ckpt,flush=True)
    # Construct the exact original architecture, including all SCAD pretrained weights;
    # then overwrite ALL weights using the fine-tuned source-dataset checkpoint.
    model,_=mod.build_model(name,args.sources)
    saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    if not isinstance(saved,dict) or 'model' not in saved:
        raise ValueError(f'Unexpected best.pt format: {ckpt}')
    incompat=model.load_state_dict(saved['model'],strict=True)
    print('Fine-tuned checkpoint loaded STRICTLY. Best epoch:',saved.get('epoch','unknown'),flush=True)
    del saved
    model=model.cuda().eval()
    ds=load_target_cache(args.dadmod,args.ccdmod,entries,target,cfg,args)
    batch=min(args.batch, 2 if name=='vjepa2_1_l' else (4 if name=='videomae_v1_l' else args.batch))
    loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=args.workers,
                      pin_memory=True,drop_last=False,
                      **({'multiprocessing_context':'fork','prefetch_factor':2} if args.workers else {}))
    result,y,scores,preds=eval_once(mod,model,cfg,loader,entries)
    result.update(source_dataset=origin,target_dataset=target,model=name,seed=42,
                  checkpoint=str(ckpt),fine_tuning_on_target=False,
                  checkpoint_selected_on_target=False)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['video','label','accident_probability','predicted_label'])
        for (_,_,rel),yy,pr,yp in zip(entries,y,scores,preds):
            writer.writerow([rel,yy,f'{pr:.10f}',yp])
    print('CROSS TEST',json.dumps({k:v for k,v in result.items() if k not in ('checkpoint',)}),flush=True)
    del loader,ds,model
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description='Seed-42 DAD/CCD cross-dataset inference, no training')
    base=Path('/home/user/22104053')
    p.add_argument('--base',type=Path,default=base)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--only',choices=['both','dad_to_ccd','ccd_to_dad'],default='both')
    p.add_argument('--continue-on-error',action='store_true')
    args=p.parse_args()
    args.dad=args.base/'DAD';args.ccd=args.base/'CCD'
    args.out=args.base/'cross_dataset_results'
    args.sources=Path.home()/'.cache/scad_variants/sources'
    args.out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(min(2,os.cpu_count() or 2))
    import cv2
    cv2.setNumThreads(1)
    # Load both scripts but NEVER run their CLI training entrypoints.
    dadmod=import_script(args.base/'train_dad_frontier_fast_ram.py','dad_frontier_original')
    ccdmod=import_script(args.base/'train_ccd_videomaev2_s_ram.py','ccd_v2s_original')
    args.dadmod=dadmod;args.ccdmod=ccdmod
    ccd_test=read_ccd_test(args.ccd)
    dad_test=read_dad_test(args.dad,dadmod)
    print('Target test distributions: CCD',len(ccd_test),'positive',sum(x[1] for x in ccd_test),
          '| DAD',len(dad_test),'positive',sum(x[1] for x in dad_test),flush=True)
    tasks=[]
    if args.only in ('both','dad_to_ccd'):
        tasks += [(name,'DAD','CCD',dadmod,ccd_test) for name in DAD_MODELS]
    if args.only in ('both','ccd_to_dad'):
        tasks += [('videomaev2_s','CCD','DAD',ccdmod,dad_test)]
    results=[];failures=[]
    for name,origin,target,mod,entries in tasks:
        try:
            results.append(run_one(name,origin,target,mod,args,entries))
        except Exception as exc:
            print(f'FAILED {origin}->{target} {name}: {exc}',flush=True)
            traceback.print_exc()
            failures.append({'model':name,'source':origin,'target':target,'error':str(exc)})
            if not args.continue_on_error:raise
    if results:
        with (args.out/'cross_dataset_seed42_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['source_dataset','target_dataset','model','seed','n_test','accuracy','macro_f1','accident_f1','video_ap','roc_auc'])
            writer.writeheader()
            for x in results:writer.writerow({k:x[k] for k in writer.fieldnames})
    if failures:
        (args.out/'failures.json').write_text(json.dumps(failures,indent=2))
        raise SystemExit(f'{len(failures)} evaluations failed; see {args.out}/failures.json')
    print('\nFinished',len(results),'cross-dataset evaluations. Summary:',args.out/'cross_dataset_seed42_summary.csv',flush=True)

if __name__=='__main__':main()
