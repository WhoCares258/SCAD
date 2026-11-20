#!/usr/bin/env python3
"""Batched, uncached ACCIDENT evaluation for original SCAD checkpoints.

Place beside scad_external_eval.py and existing SCAD scripts.
Run on a CUDA GPU compute node. No training and no disk frame cache.
Labels inferred from accident timestamps remain provisional.
"""
from __future__ import annotations
import argparse, csv, gc, json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

import scad_external_eval as base


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--home',type=Path,default=Path('/home/user/22104053'))
    p.add_argument('--model',choices=base.MODEL_NAMES,required=True)
    p.add_argument('--seed',type=int,choices=range(42,47),required=True)
    p.add_argument('--batch-size',type=int,default=8,help='Initial batch size, automatically reduced on CUDA OOM')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--report-every',type=int,default=100)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--no-auto-batch',action='store_true')
    a=p.parse_args()
    if a.batch_size<1 or a.report_every<1 or a.limit<0 or a.offset<0: p.error('invalid batch/limit/offset')
    return a


def metrics(rows):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    if not rows:return {}
    y=np.array([int(r['label']) for r in rows]); pred=np.array([int(r['prediction']) for r in rows]); probs=np.array([float(r['prob_accident']) for r in rows])
    cm=confusion_matrix(y,pred,labels=[0,1]).tolist()
    return {'evaluated':len(rows),'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'accident_recall':float(cm[1][1]/sum(cm[1])) if sum(cm[1]) else None,
            'normal_specificity':float(cm[0][0]/sum(cm[0])) if sum(cm[0]) else None,
            'auc':float(roc_auc_score(y,probs)) if len(set(y))==2 else None,
            'confusion_matrix_normal_accident':cm}


def release_cuda(torch):
    gc.collect()
    torch.cuda.empty_cache()


def main():
    a=arguments(); a.home=a.home.resolve();sys.path.insert(0,str(a.home))
    import torch
    import PyNvVideoCodec as nvc
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU allocation is required, not the login node')
    torch.set_num_threads(min(2,os.cpu_count() or 2))
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    full=base.list_windows(a.home)
    if a.offset>=len(full):raise ValueError('offset beyond manifest')
    rows=full[a.offset:a.offset+a.limit if a.limit else None]
    suffix=f'_offset{a.offset}_limit{a.limit}' if a.offset or a.limit else ''
    root=(a.out or a.home/'scad/external_accident/batched_results').resolve()
    folder=root/a.model/f'seed_{a.seed}';folder.mkdir(parents=True,exist_ok=True)
    csvpath=folder/f'predictions{suffix}.csv';jsonpath=folder/f'summary{suffix}.json'
    if (csvpath.exists() or jsonpath.exists()) and not a.overwrite:
        raise FileExistsError(f'Output exists; explicitly use --overwrite: {csvpath}')
    model,cfg,checkpoint=base.load_checkpoint(a.home,a.model,a.seed)
    amp=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    mean=torch.tensor(cfg['mean'],device='cuda',dtype=torch.float32).view(1,1,3,1,1)
    std=torch.tensor(cfg['std'],device='cuda',dtype=torch.float32).view(1,1,3,1,1)
    print(f'MODEL={a.model} SEED={a.seed} CHECKPOINT={checkpoint}',flush=True)
    print(f'INPUT={cfg["frames"]}f {cfg["width"]}x{cfg["height"]}; AMP={amp}; windows={len(rows)}; batch_start={a.batch_size}; NO CACHE',flush=True)
    fps_cache={}; good=[]; failures=[]; batch_actual=[]
    fieldnames=['window_id','source_id','video_path','category','start','end','impact_time','label',
                'prediction','prob_accident','correct','decode_preprocess_ms','model_ms','e2e_ms','error']
    since=time.perf_counter();completed=0;next_report=a.report_every;batch_size=a.batch_size
    torch.cuda.reset_peak_memory_stats()
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader(); f.flush()
        i=0
        while i<len(rows):
            selected=rows[i:i+batch_size]
            prepared=[]; input_gpu=None; logits=None; errors=[]
            try:
                torch.cuda.synchronize();t0=time.perf_counter()
                for r in selected:
                    path=r['video_path']
                    if path not in fps_cache:fps_cache[path]=base.query_video_info(path)
                    prepared.append(base.decoded_window(path,r['start'],r['end'],cfg,fps_cache[path],nvc,torch))
                x=torch.stack(prepared,0).permute(0,1,4,2,3).contiguous()
                # Original SCAD forward_batch: uint8 -> float/255 -> cfg normalization -> B,C,T,H,W
                input_gpu=((x.float().div_(255.0)-mean)/std).permute(0,2,1,3,4).contiguous()
                torch.cuda.synchronize();t1=time.perf_counter()
                with torch.inference_mode(),torch.autocast('cuda',dtype=amp):
                    logits=model(input_gpu)
                    if tuple(logits.shape)!=(len(selected),2):
                        raise RuntimeError(f'Expected logits [{len(selected)},2], got {tuple(logits.shape)}')
                    prob=logits.float().softmax(-1)[:,1].cpu().tolist()
                torch.cuda.synchronize();t2=time.perf_counter()
                decode_ms=(t1-t0)*1000/len(selected)
                model_ms=(t2-t1)*1000/len(selected)
                e2e_ms=(t2-t0)*1000/len(selected)
                for r,p in zip(selected,prob):
                    pred=int(p>=0.5)
                    row={k:r[k] for k in ('window_id','source_id','video_path','category','start','end','impact_time','label')}
                    row.update(prediction=pred,prob_accident=f'{p:.10f}',correct=int(pred==r['label']),
                               decode_preprocess_ms=round(decode_ms,4),model_ms=round(model_ms,4),
                               e2e_ms=round(e2e_ms,4),error='')
                    good.append(row);writer.writerow(row)
                i+=len(selected);completed=i;batch_actual.append(len(selected))
                del x,prepared,input_gpu,logits
            except torch.cuda.OutOfMemoryError:
                del prepared,input_gpu,logits
                release_cuda(torch)
                if a.no_auto_batch or batch_size==1:raise
                batch_size=max(1,batch_size//2)
                print(f'[OOM] Retrying from window {i} with smaller batch={batch_size}',flush=True)
                continue
            except Exception as exc:
                # Fail closed: do not turn failed decoding/inference into a valid score.
                error=f'{type(exc).__name__}: {exc}'
                print(f'ERROR window {i} batch={len(selected)}: {error}',flush=True)
                for r in selected:
                    row={k:r[k] for k in ('window_id','source_id','video_path','category','start','end','impact_time','label')}
                    row.update(prediction='',prob_accident='',correct='',decode_preprocess_ms='',model_ms='',e2e_ms='',error=error)
                    writer.writerow(row);failures.append(row)
                i+=len(selected);completed=i
                del prepared,input_gpu,logits
            if completed>=next_report or completed==len(rows):
                f.flush()
                m=metrics(good)
                elapsed=time.perf_counter()-since
                fmt=lambda v:'n/a' if v is None else f'{v*100:.2f}%'
                print(f'RUNNING {completed}/{len(rows)} valid={len(good)} errors={len(failures)} '
                      f'acc={fmt(m.get("accuracy"))} f1={fmt(m.get("macro_f1"))} '
                      f'accident_recall={fmt(m.get("accident_recall"))} '
                      f'normal_specificity={fmt(m.get("normal_specificity"))} '
                      f'avg_clips_s={completed/max(elapsed,1e-9):.2f} current_batch={batch_size}',flush=True)
                next_report=completed+a.report_every
    if not good:raise RuntimeError('No valid predictions produced')
    result=metrics(good)
    type_recalls={}
    for typ in sorted(set(r['category'] for r in good if int(r['label'])==1)):
        s=[r for r in good if int(r['label'])==1 and r['category']==typ]
        type_recalls[typ]={'n':len(s),'recall':sum(int(z['prediction']) for z in s)/len(s)}
    result.update(model=a.model,seed=a.seed,checkpoint=str(checkpoint),
                  status='PROVISIONAL: timestamp labels; no manual audit or SCAD source-overlap exclusion',
                  pipeline='Original NVDEC RGBP indexed frame sampling, GPU letterbox, cfg normalization; direct 4s windows',
                  gpu=torch.cuda.get_device_name(0),amp=str(amp),
                  requested=len(rows),errors=len(failures),batch_size_requested=a.batch_size,
                  batch_size_final=batch_size,batch_size_used_min=min(batch_actual) if batch_actual else None,
                  frames=int(cfg['frames']),width=int(cfg['width']),height=int(cfg['height']),
                  peak_allocated_vram_mib=torch.cuda.max_memory_allocated()/2**20,
                  elapsed_s=time.perf_counter()-since,clips_per_second=len(good)/max(time.perf_counter()-since,1e-9),
                  accident_recall_by_type=type_recalls,predictions_csv=str(csvpath))
    for key in ('decode_preprocess_ms','model_ms','e2e_ms'):
        vals=np.array([float(r[key]) for r in good]);result[key]={
            'mean':float(vals.mean()),'p50':float(np.percentile(vals,50)),'p95':float(np.percentile(vals,95))}
    jsonpath.write_text(json.dumps(result,indent=2))
    print('\nFINAL',json.dumps(result,indent=2),flush=True)
    if failures:raise RuntimeError(f'{len(failures)} failed windows; results incomplete: {csvpath}')

if __name__=='__main__':main()
