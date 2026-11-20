#!/usr/bin/env python3
"""SCAD 10-configuration NVDEC+cache five-seed campaign, L40S.

This launches *original* training functions using exactly their model builders,
optimizers, schedulers, augmentation, batch probing, checkpoint selection and
one final test evaluation. Only CachedVideos is replaced by a versioned
NVDEC/GPU-letterbox uint8 cache. Every training job executes in its own process.

Required: PyTorch/CUDA, numpy, scikit-learn, PyNvVideoCodec, ffmpeg/ffprobe,
original seven research runners in scad_17_sources/ (bundled).
No CUDA training was performed by the author of this wrapper.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

VERSION='scad_final10_nvdec_uint8_gpu_letterbox_v1'
EXPECTED={'train':778,'val':112,'test':232}
SPECS={
  'videomae_v1_l':('scad_final_nine_l40s.py','videomae_v1_l'),
  'videomaev2_b':('scad_six_l40s.py','videomaev2_b'),
  'videomaev2_s':('scad_variants_fixed_v2.py','videomaev2_s'),
  'vjepa21_l':('scad_variants_fixed_v2.py','vjepa21_l'),
  'vjepa21_b_full':('scad_next_seven_no_flash_v2.py','vjepa21_full'),
  'internvideo2_l':('scad_variants_fixed_v2.py','internvideo2_l'),
  'internvideo2_b':('scad_next_seven_no_flash_v2.py','internvideo2_b'),
  'internvideo2_s':('scad_final_nine_l40s.py','internvideo2_s'),
  'mvitv2_b':('scad_variants_fixed_v2.py','mvitv2_b'),
  'mvitv2_s':('scad_six_l40s.py','mvitv2_s'),
}
DEFAULT_HOME=Path('/home/user/22104053')
SOURCE_CACHE_GROUP={
  'scad_six_l40s.py':'scad_six',
  'scad_next_seven_no_flash_v2.py':'scad_next',
  'scad_variants_fixed_v2.py':'scad_variants',
  'scad_final_nine_l40s.py':'scad_variants',
  'scad_videomae_v1_sb.py':'scad_variants',
}

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=str))
    os.replace(tmp,path)

def source_file(model):
    filename=SPECS[model][0]
    search=[Path(__file__).resolve().parent/'scad_17_sources'/filename,
            Path(__file__).resolve().parent/filename,
            DEFAULT_HOME/'scad_17_sources'/filename,
            DEFAULT_HOME/filename]
    for file in search:
        if file.is_file():return file
    raise FileNotFoundError(f'Missing original model runner {filename}; extract ZIP completely')

def source_module(model):
    file=source_file(model)
    name='_scad_nvdec_original_'+file.stem
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,file)
        mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod
        spec.loader.exec_module(mod)
    mod=sys.modules[name]
    inner=SPECS[model][1]
    if inner not in mod.SPECS:raise RuntimeError(f'{file} lacks spec for {inner}')
    return mod,inner

def source_dir(model,args):
    filename=SPECS[model][0]
    return args.home/'.cache'/SOURCE_CACHE_GROUP[filename]/'sources'

def scan_splits(args):
    mod,_=source_module('videomaev2_s')
    result={}
    for split,n in EXPECTED.items():
        rows=mod.scan(args.data,split)
        if len(rows)!=n:raise RuntimeError(f'{split}: expected {n} clips, found {len(rows)}')
        if any(y not in (0,1) for _,y in rows):raise RuntimeError(f'Invalid labels in {split}')
        labels={str(y):sum(label==y for _,label in rows) for y in (0,1)}
        if split=='test' and labels!={'0':116,'1':116}:
            raise RuntimeError(f'Test class counts are not 116/116: {labels}')
        result[split]=rows
    paths=[str(Path(path).resolve()) for split in EXPECTED for path,_ in result[split]]
    if len(paths)!=len(set(paths)):raise RuntimeError('Duplicate file paths across train/val/test')
    return result

def run_checked(cmd):
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if r.returncode:raise RuntimeError(f'Command returned {r.returncode}: {cmd}\n{r.stderr[-2000:]}')
    return r.stdout

def probe(path):
    data=json.loads(run_checked(['ffprobe','-v','error','-select_streams','v:0',
             '-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_frames',
             '-of','json',str(path)]))['streams'][0]
    return {k:data.get(k) for k in ['codec_name','width','height','avg_frame_rate','nb_frames']}

def opencv_count(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError('Cannot read video with OpenCV: '+str(path))
    try:
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n<1:
            n=0
            while cap.grab():n+=1
        if n<1:raise RuntimeError('Zero frames: '+str(path))
        return n
    finally:cap.release()

def gpu_decode(path,num_frames):
    """GPU-resident RGBP frame indices, original frame count and live references."""
    import PyNvVideoCodec as nvc
    import numpy as np
    import torch
    decoder=nvc.SimpleDecoder(str(path),gpu_id=0,use_device_memory=True,
             output_color_type=nvc.OutputColorType.RGBP,
             need_scanned_stream_metadata=True)
    n=len(decoder)
    if n<=0:raise RuntimeError('NVDEC zero frames: '+str(path))
    indices=np.linspace(0,n-1,int(num_frames)).round().astype(int).tolist()
    unique=list(dict.fromkeys(indices))
    frames=decoder.get_batch_frames_by_index(unique)
    if len(frames)!=len(unique):raise RuntimeError(f'Decoder emitted {len(frames)} of {len(unique)} indexed frames')
    tensors=[torch.from_dlpack(f) for f in frames]
    if any(t.device.type!='cuda' or t.ndim!=3 or t.shape[0]!=3 or t.dtype!=torch.uint8 for t in tensors):
        raise RuntimeError('Expected CUDA uint8 RGBP frames [3,H,W]; refusing CPU fallback')
    by_index={idx:i for i,idx in enumerate(unique)}
    rgb=torch.stack([tensors[by_index[idx]] for idx in indices],0)
    return rgb,n,decoder,frames

def gpu_letterbox_uint8(rgb,width,height):
    """Output uint8 [T,H,W,3], matching the existing NVDEC pipeline's GPU resize.

    Its inference pipeline first resizes float tensors, rounds to pixel range,
    letterboxes, then normalizes. We cache the uint8 intermediate and apply the
    SAME normalizations/flip in the original trainer's forward_batch.
    """
    import torch
    import torch.nn.functional as F
    if rgb.ndim!=4 or rgb.shape[1]!=3:raise ValueError('Expected [T,3,H,W]')
    _,_,ih,iw=rgb.shape
    width,height=int(width),int(height)
    scale=min(width/iw,height/ih)
    nw=min(width,max(1,round(iw*scale)))
    nh=min(height,max(1,round(ih*scale)))
    x=rgb.float()
    if (iw,ih)!=(nw,nh):
        if scale<1:x=F.interpolate(x,size=(nh,nw),mode='area')
        else:x=F.interpolate(x,size=(nh,nw),mode='bilinear',align_corners=False)
        x=x.round().clamp_(0,255)
    left=(width-nw)//2;top=(height-nh)//2
    if (nw,nh)!=(width,height):
        x=F.pad(x,(left,width-nw-left,top,height-nh-top),mode='constant',value=0)
    x=x.to(torch.uint8).permute(0,2,3,1).contiguous()
    if tuple(x.shape[-3:])!=(height,width,3):raise RuntimeError('Incorrect GPU letterbox shape')
    return x

def transcode(path,dest,crf):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_name(dest.stem+'.building.mp4')
    try:
        run_checked(['ffmpeg','-hide_banner','-nostdin','-loglevel','error','-y',
            '-i',str(path),'-map','0:v:0','-an','-sn','-dn','-c:v','libx264',
            '-crf',str(crf),'-preset','medium','-pix_fmt','yuv420p',
            '-fps_mode','passthrough','-movflags','+faststart',str(temp)])
        if not temp.exists() or temp.stat().st_size==0:raise RuntimeError('Empty transcode')
        os.replace(temp,dest)
    finally:
        if temp.exists():temp.unlink()

def manifest_path(args):return args.out/'prepared_manifest.json'

def manifest_read(args,full_validate=False):
    path=manifest_path(args)
    if not path.exists():raise RuntimeError('Missing prepared manifest; run --stage prepare')
    obj=json.loads(path.read_text())
    if obj.get('version')!=VERSION or obj.get('dataset')!=str(args.data.resolve()) or obj.get('crf')!=args.crf:
        raise RuntimeError('Dataset/preprocessing version differs from manifest; use a NEW --out directory')
    if len(obj.get('records',[]))!=sum(EXPECTED.values()):raise RuntimeError('Incomplete manifest')
    if any(sum(rec['split']==s for rec in obj['records'])!=n for s,n in EXPECTED.items()):
        raise RuntimeError('Incorrect manifest split sizes')
    if full_validate:
        current=scan_splits(args)
        matched={(r['split'],r['original']):r for r in obj['records']}
        for split,rows in current.items():
            for original,label in rows:
                key=(split,str(Path(original).resolve()))
                if key not in matched:raise RuntimeError(f'Manifest does not contain {key}')
                rec=matched[key]
                src=Path(original); prepared=Path(rec['prepared'])
                if rec['label']!=label or rec['original_size']!=src.stat().st_size:
                    raise RuntimeError('Original dataset changed: '+str(src))
                if not prepared.is_file() or prepared.stat().st_size!=rec['prepared_size']:
                    raise RuntimeError('Prepared file missing/changed: '+str(prepared))
    return obj

def prepare(args):
    require_env(need_gpu=True,need_ffmpeg=True)
    if manifest_path(args).exists():
        manifest=manifest_read(args,full_validate=True)
        print('REUSING VERIFIED PREPARATION:',len(manifest['records']),'clips; converted',manifest['transcoded'],flush=True)
        return manifest
    splits=scan_splits(args)
    records=[];converted=0
    dest_dir=args.out/'h264_prepared'
    for split,items in splits.items():
        for i,(original,label) in enumerate(items):
            src=Path(original).resolve()
            before=probe(src);source_count=opencv_count(src)
            try:
                gpu,n,decoder,frames=gpu_decode(src,32)
                import torch
                torch.cuda.synchronize()
                del gpu,decoder,frames
                if n!=source_count:raise RuntimeError(f'Original/NVDEC frame counts differ: {source_count}/{n}')
                effective=src; reason='original_nvdec'
            except Exception as err:
                fingerprint=hashlib.sha256(f'{src}|{src.stat().st_size}|{src.stat().st_mtime_ns}'.encode()).hexdigest()[:14]
                effective=dest_dir/split/f'{i:04d}_{fingerprint}.mp4'
                if not effective.is_file():
                    print(f'TRANSCODE {split} {i+1}/{len(items)} {src.name}: {str(err)[:110]}',flush=True)
                    transcode(src,effective,args.crf)
                after=probe(effective)
                if (before['width'],before['height'])!=(after['width'],after['height']):
                    raise RuntimeError('Transcode changed video resolution: '+str(src))
                converted_count=opencv_count(effective)
                gpu,n,decoder,frames=gpu_decode(effective,32)
                import torch
                torch.cuda.synchronize()
                del gpu,decoder,frames
                if source_count!=n or converted_count!=n:
                    raise RuntimeError(f'Transcode frame count changed for {src}: {source_count}, {converted_count}, {n}')
                converted+=1;reason='transcoded_h264'
            records.append({'split':split,'index':i,'label':int(label),
                'original':str(src),'prepared':str(effective),'processing':reason,
                'original_size':src.stat().st_size,'prepared_size':effective.stat().st_size,
                'source_frames':source_count,'source_resolution':[before['width'],before['height']],
                'source_codec':before['codec_name']})
            if (i+1)%100==0 or i+1==len(items):
                print(f'PREPARED {split}: {i+1}/{len(items)} | converted total {converted}',flush=True)
    manifest={'version':VERSION,'dataset':str(args.data.resolve()),'crf':args.crf,
        'compatibility':'GPU NVDEC RGBP; offline H.264 CRF conversion only on unsupported source streams; no spatial resizing',
        'splits':EXPECTED,'transcoded':converted,'records':records}
    atomic_json(manifest_path(args),manifest)
    return manifest_read(args,full_validate=True)

def require_env(need_gpu=False,need_ffmpeg=False):
    if need_ffmpeg:
        for binname in ('ffmpeg','ffprobe'):
            if not shutil.which(binname):raise RuntimeError(f'{binname} not found. Activate your venv with FFmpeg symlinks.')
    if need_gpu:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA GPU is required; cannot run here')
        import PyNvVideoCodec as nvc
        print('DEVICE',torch.cuda.get_device_name(0),'PyNvVideoCodec',getattr(nvc,'__version__','installed'),flush=True)

def make_nvdec_dataset(mod,manifest,args):
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    records={(r['split'],r['original']):r for r in manifest['records']}
    manifest_digest=hashlib.sha256(manifest_path(args).read_bytes()).hexdigest()
    class NVDECCachedVideos(Dataset):
        def __init__(self,items,cfg,cache_dir,split,ram=True):
            self.items=items;self.split=split
            folder=Path(cache_dir);folder.mkdir(parents=True,exist_ok=True)
            key=f'{split}_{cfg["frames"]}f_{cfg["width"]}x{cfg["height"]}_{VERSION}'
            self.path=folder/(key+'.npy')
            self.meta=folder/(key+'.json')
            paths=[]
            for original,label in items:
                rec=records.get((split,str(Path(original).resolve())))
                if rec is None or rec['label']!=label:raise RuntimeError(f'Manifest label/path mismatch: {original}')
                paths.append(rec)
            signature={'pipeline':VERSION,'manifest_sha256':manifest_digest,
                'paths':[[r['original'],r['original_size'],r['prepared_size'],r['label']] for r in paths],
                'frames':cfg['frames'],'width':cfg['width'],'height':cfg['height']}
            can_reuse=self.path.exists() and self.meta.exists()
            if can_reuse:
                try:can_reuse=json.loads(self.meta.read_text())==signature
                except Exception:can_reuse=False
            if not can_reuse:
                tmp=self.path.with_name(self.path.stem+'.building.npy')
                shape=(len(paths),int(cfg['frames']),int(cfg['height']),int(cfg['width']),3)
                print('NVDEC CACHE CREATE',split,shape,'at',self.path,flush=True)
                try:
                    mmap=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=shape)
                    for i,rec in enumerate(paths):
                        gpu,n,decoder,frames=gpu_decode(rec['prepared'],cfg['frames'])
                        if n!=rec['source_frames']:raise RuntimeError('Frame drift '+rec['original'])
                        rgb8=gpu_letterbox_uint8(gpu,cfg['width'],cfg['height'])
                        mmap[i]=rgb8.cpu().numpy()
                        del gpu,n,decoder,frames,rgb8
                        if (i+1)%100==0 or i+1==len(paths):
                            print('  GPU NVDEC cached',i+1,'/',len(paths),'for',split,flush=True)
                    mmap.flush();del mmap
                    os.replace(tmp,self.path)
                    atomic_json(self.meta,signature)
                finally:
                    if tmp.exists():tmp.unlink()
            else:
                print('NVDEC CACHE REUSE',split,self.path,flush=True)
            self.arr=np.load(self.path,mmap_mode='r')
            expected=(len(paths),int(cfg['frames']),int(cfg['height']),int(cfg['width']),3)
            if self.arr.shape!=expected or self.arr.dtype!=np.uint8:
                raise RuntimeError('Invalid NVDEC cache shape/dtype; use a fresh --out folder')
        def __len__(self):return len(self.items)
        def __getitem__(self,i):
            x=torch.from_numpy(np.array(self.arr[i],copy=True)).permute(0,3,1,2)
            return x,int(self.items[i][1])
    mod.CachedVideos=NVDECCachedVideos

def capture_test_predictions(mod,run_dir,model_name,seed):
    """Capture predictions during ORIGINAL eval_model's ONE test evaluation pass."""
    import torch
    from sklearn.metrics import f1_score
    original_eval=mod.eval_model
    original_forward=mod.forward_batch
    def intercepted_eval(model,loader,cfg,amp_dtype):
        if getattr(loader.dataset,'split',None)!='test':
            return original_eval(model,loader,cfg,amp_dtype)
        probs=[]
        def intercepted_forward(*a,**kw):
            logits,labels=original_forward(*a,**kw)
            probs.extend(logits.detach().float().softmax(-1)[:,1].cpu().tolist())
            return logits,labels
        mod.forward_batch=intercepted_forward
        try:result=original_eval(model,loader,cfg,amp_dtype)
        finally:mod.forward_batch=original_forward
        items=loader.dataset.items
        if len(probs)!=len(items) or len(items)!=EXPECTED['test']:
            raise RuntimeError('Test prediction count mismatch; cannot certify run')
        rows=[]
        for i,((path,label),prob) in enumerate(zip(items,probs)):
            rows.append({'index':i,'video':str(Path(path).resolve()),'label':label,
                         'prob_accident':format(prob,'.10f'),'prediction':int(prob>=.5),
                         'correct':int(int(prob>=.5)==label)})
        with (run_dir/'test_predictions.csv').open('w',newline='') as fh:
            wr=csv.DictWriter(fh,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
        ys=[r['label'] for r in rows];yp=[r['prediction'] for r in rows]
        macro=f1_score(ys,yp,average='macro',zero_division=0)*100
        if abs(100*sum(y==p for y,p in zip(ys,yp))/len(ys)-result['accuracy'])>0.001:
            raise RuntimeError('Original evaluation accuracy disagrees with captured predictions')
        atomic_json(run_dir/'test_classwise.json',{'macro_f1_pct':macro,
                    'normal_recall_pct':100*sum(y==0 and p==0 for y,p in zip(ys,yp))/ys.count(0),
                    'accident_recall_pct':100*sum(y==1 and p==1 for y,p in zip(ys,yp))/ys.count(1),
                    'correct':sum(y==p for y,p in zip(ys,yp)),'n':len(ys)})
        return result
    mod.eval_model=intercepted_eval

def result_path(model,seed,args):return args.out/'runs'/model/f'seed_{seed}'/'final10_complete.json'

def validate_run(model,seed,args):
    p=result_path(model,seed,args)
    if not p.exists():return None
    r=json.loads(p.read_text())
    if (r.get('model')!=model or r.get('seed')!=seed or r.get('epochs')!=args.epochs
       or r.get('pipeline')!=VERSION):
        raise RuntimeError(f'Existing completed run has mismatched identity/settings: {p}')
    base=p.parent
    if not (base/'test_predictions.csv').exists() or not (base/'test_classwise.json').exists():
        raise RuntimeError(f'Run completion marker present but missing prediction files: {base}')
    return r

def run_one(args,model,seed):
    require_env(need_gpu=True)
    manifest=manifest_read(args,full_validate=True)
    previous=validate_run(model,seed,args)
    if previous:
        print('SKIP VERIFIED COMPLETE',model,'seed',seed,flush=True)
        return previous
    # Recompute after the job's own pre-run, to prevent accidental test leak.
    mod,inner=source_module(model)
    make_nvdec_dataset(mod,manifest,args)
    run_dir=args.out/'runs'/model/f'seed_{seed}'
    run_dir.mkdir(parents=True,exist_ok=True)
    capture_test_predictions(mod,run_dir,model,seed)
    requested=SimpleNamespace(out=run_dir,cache=args.out/'frame_cache',
        sources=source_dir(model,args),epochs=args.epochs,seed=seed,
        workers=args.workers,autobatch=True,rerun=True)
    import torch
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    os.environ.setdefault('OMP_NUM_THREADS','2')
    torch.set_num_threads(min(2,os.cpu_count() or 2))
    print('START TRAINING',model,'seed',seed,'original runner',source_file(model),flush=True)
    result=mod.train(inner,args.data,requested)
    if not isinstance(result,dict) or not math.isfinite(float(result['test_accuracy'])):
        raise RuntimeError('Trainer did not return valid test metrics')
    if not (0<=result['test_accuracy']<=100):raise RuntimeError('Invalid accuracy')
    if result.get('train_samples')!=778 or result.get('val_samples')!=112:
        raise RuntimeError('Original runner used wrong train/val size')
    epochs_files=list(run_dir.glob('*_epochs.json'))
    ckpts=list(run_dir.glob('*_best.pt'))
    if len(epochs_files)!=1 or len(ckpts)!=1:raise RuntimeError('Missing/ambiguous history or best checkpoint')
    epoch_history=json.loads(epochs_files[0].read_text())
    if len(epoch_history)!=args.epochs:raise RuntimeError('Incorrect number of training epochs')
    for epoch in epoch_history:
        for k in ('train_loss','val_loss','val_accuracy'):
            if not math.isfinite(float(epoch[k])):raise RuntimeError('Non-finite '+k)
    classwise=json.loads((run_dir/'test_classwise.json').read_text())
    record={'pipeline':VERSION,'model':model,'inner_model':inner,'seed':seed,
        'epochs':args.epochs,'frames':mod.SPECS[inner]['frames'],
        'width':mod.SPECS[inner]['width'],'height':mod.SPECS[inner]['height'],
        'checkpoint':str(ckpts[0]),'source_script':str(source_file(model)),
        'best_epoch':result['best_epoch'],'validation_accuracy_pct':result['best_val_accuracy'],
        'test_accuracy_pct':result['test_accuracy'],'test_f1_pct':result['test_f1'],
        'test_macro_f1_pct':classwise['macro_f1_pct'],
        'normal_recall_pct':classwise['normal_recall_pct'],
        'accident_recall_pct':classwise['accident_recall_pct'],
        'test_confusion_matrix':result['confusion_matrix'],
        'training_wall_seconds':result.get('wall_seconds'),
        'data_policy':'NVDEC GPU-resized uint8 memmap cached once; original trainer normalizations and online flip',
        'selection_policy':'best validation accuracy (tie by val loss); fixed 232 test clips evaluated once per seed',
        'run_folder':str(run_dir)}
    atomic_json(result_path(model,seed,args),record)
    print('CERTIFIED COMPLETE',json.dumps({'model':model,'seed':seed,
        'acc':record['test_accuracy_pct'],'macro_f1':record['test_macro_f1_pct']}),flush=True)
    return record

def job_command(args,model,seed):
    return [sys.executable,'-u',str(Path(__file__).resolve()),'--stage','run-one',
        '--model',model,'--seed',str(seed),'--epochs',str(args.epochs),
        '--crf',str(args.crf),'--workers',str(args.workers),'--home',str(args.home),
        '--data',str(args.data),'--out',str(args.out)]

def run_child(args,model,seed):
    if validate_run(model,seed,args):
        print('SKIP',model,seed,'(already completed)',flush=True);return
    folder=args.out/'logs';folder.mkdir(parents=True,exist_ok=True)
    log=folder/f'{model}_seed{seed}.log'
    print('\n>>> RUN',model,'seed',seed,'log',log,flush=True)
    with log.open('a',buffering=1) as fh:
        fh.write('\n=== START OR RESUME '+time.strftime('%Y-%m-%d %H:%M:%S')+' ===\n')
        process=subprocess.Popen(job_command(args,model,seed),stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=args.home)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);fh.write(line)
            code=process.wait()
        except KeyboardInterrupt:
            process.terminate();process.wait();raise
    if code:raise RuntimeError(f'Run failed: {model} seed{seed}; exit={code}; see {log}')
    if not validate_run(model,seed,args):
        raise RuntimeError('Child process exited successfully without completion marker: '+str(log))

def summary(args):
    seeds=parse_seeds(args.seeds)
    rows=[]
    for model in SPECS:
        for seed in seeds:
            r=validate_run(model,seed,args)
            if r:rows.append(r)
    args.out.mkdir(parents=True,exist_ok=True)
    fields=['model','seed','epochs','best_epoch','validation_accuracy_pct','test_accuracy_pct',
      'test_f1_pct','test_macro_f1_pct','normal_recall_pct','accident_recall_pct','training_wall_seconds']
    with (args.out/'final10_all_runs.csv').open('w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=fields,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
    bymodel=[]
    import statistics
    for model in SPECS:
        these=[r for r in rows if r['model']==model]
        if not these:continue
        vals=[r['test_accuracy_pct'] for r in these]
        f1=[r['test_macro_f1_pct'] for r in these]
        bymodel.append({'model':model,'n_seeds':len(these),
            'mean_accuracy_pct':round(statistics.mean(vals),4),
            'sd_accuracy_pct':round(statistics.stdev(vals),4) if len(vals)>1 else None,
            'mean_macro_f1_pct':round(statistics.mean(f1),4),
            'sd_macro_f1_pct':round(statistics.stdev(f1),4) if len(f1)>1 else None})
    with (args.out/'final10_mean_sd.csv').open('w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=['model','n_seeds','mean_accuracy_pct','sd_accuracy_pct',
             'mean_macro_f1_pct','sd_macro_f1_pct']);wr.writeheader();wr.writerows(bymodel)
    print('SUMMARY',len(rows),'/',len(SPECS)*len(seeds),'runs completed',flush=True)
    for r in bymodel:
        print(f"{r['model']:<23} n={r['n_seeds']} mean acc={r['mean_accuracy_pct']:.3f} sd={r['sd_accuracy_pct']}",flush=True)
    print('CSV:',args.out/'final10_all_runs.csv',args.out/'final10_mean_sd.csv',flush=True)

def parse_seeds(value):
    nums=[int(x.strip()) for x in value.split(',')]
    if len(nums)!=5 or len(set(nums))!=5:raise ValueError('Require five distinct seed values, e.g. 42,43,44,45,46')
    return nums

def selftest():
    assert len(SPECS)==10 and len(set(SPECS))==10
    assert sum(EXPECTED.values())==1122
    assert len(set(x for x,_ in SPECS.values()))==4
    for model in SPECS:
        mod,inner=source_module(model)
        cfg=mod.SPECS[inner]
        assert cfg['frames'] in (8,16,32) and cfg['width'] in (224,384)
        assert callable(mod.train) and callable(mod.eval_model)
        print('SOURCE OK',model,'->',SPECS[model],'cfg',cfg['frames'],cfg['width'],cfg['height'])
    print('SELFTEST PASS: all ten original model builders and training functions are present; CUDA still required for actual run.')

def main():
    p=argparse.ArgumentParser(description='SCAD 10-model x 5-seed NVDEC+cache campaign')
    p.add_argument('--stage',choices=['selftest','prepare','pilot','all','launch','run-one','summary'],required=True)
    p.add_argument('--home',type=Path,default=DEFAULT_HOME)
    p.add_argument('--data',type=Path,default=DEFAULT_HOME/'accident')
    p.add_argument('--out',type=Path,default=DEFAULT_HOME/'scad'/'nvdec_final10')
    p.add_argument('--model',choices=list(SPECS),default='videomaev2_s')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--seeds',default='42,43,44,45,46')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--crf',type=int,default=14)
    args=p.parse_args()
    args.home=args.home.resolve();args.data=args.data.resolve();args.out=args.out.resolve()
    if args.epochs<1:raise ValueError('epochs must be positive')
    if args.workers<0:raise ValueError('workers must be nonnegative')
    if not 0<=args.crf<=51:raise ValueError('CRF out of range')
    if args.stage=='selftest':selftest();return
    if args.stage=='prepare':prepare(args);return
    if args.stage=='run-one':run_one(args,args.model,args.seed);return
    if args.stage=='summary':summary(args);return
    # All three stages in a single command, with automatic stop if any stage fails.
    if args.stage=='launch':prepare(args)
    else:manifest_read(args,full_validate=True)
    if args.stage in ('pilot','launch','all'):
        seeds=parse_seeds(args.seeds)
        # Gate all 50 runs on a COMPLETE and FINITE pilot, not a test-accuracy threshold.
        pilot_seed=seeds[0]
        if args.stage in ('pilot','launch','all'):
            run_child(args,'videomaev2_s',pilot_seed)
            print('PILOT VERIFIED:',result_path('videomaev2_s',pilot_seed,args),flush=True)
        if args.stage in ('all','launch'):
            for model in SPECS:
                for seed in seeds:
                    if model=='videomaev2_s' and seed==pilot_seed:continue
                    run_child(args,model,seed)
        summary(args)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
    except Exception as exc:
        print('NVDEC FINAL10 FAILED:',exc,file=sys.stderr,flush=True)
        traceback.print_exc();raise SystemExit(2)
