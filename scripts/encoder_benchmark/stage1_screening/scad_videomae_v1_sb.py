#!/usr/bin/env python3
"""SCAD VideoMAE V1 Small + Base supervised Kinetics-400 fine-tuning.

Compatible with the successful V1-Large experiment in scad_final_nine_l40s.py:
- Official MCG-NJU HF K400-finetuned VideoMAE V1-S and V1-B checkpoints.
- Two-class classification head newly initialized; ALL backbone weights trainable.
- Exactly 16 full-video uniform frames, 224x224 full-scene RGB letterbox.
- AdamW lr 1e-5, wd .05, cosine anneal, 10 epochs, BF16/AMP, seed 42.
- Effective batch near 32 using physical batch probing + gradient accumulation.
- Highest validation accuracy (tie break: val loss); test once after selection.
- Separate scad_variant_videomae_v1_s/b_* files; earlier results retained.

Usage (L40S, existing venv):
    python -u scad_videomae_v1_sb.py --preflight-only
    python -u scad_videomae_v1_sb.py
    python -u scad_videomae_v1_sb.py --models videomae_v1_b

This script does not claim architecture-only V1 vs V2 comparisons: the starting
pretraining procedures and Kinetics-400/710 supervised stages differ.
"""
import argparse, csv, gc, json, os, random, sys, time, traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

SPECS = {
    'videomae_v1_s': dict(width=224,height=224,frames=16,batch=8,lr=1e-5,wd=.05,
                          mean=(.485,.456,.406),std=(.229,.224,.225)),
    'videomae_v1_b': dict(width=224,height=224,frames=16,batch=8,lr=1e-5,wd=.05,
                          mean=(.485,.456,.406),std=(.229,.224,.225)),
}

LABELS = {'normal':0,'non-accident':0,'non_accident':0,'nonviolence':0,'anomalous':1,'accident':1,'abnormal':1}
EXTS={'.mp4','.avi','.mov','.mkv','.mpeg','.mpg','.webm'}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def scan(data_root, split):
    base=Path(data_root)/split
    if not base.is_dir(): raise FileNotFoundError(f'Missing {base}. Expected train/ val/ test/ class subfolders')
    items=[]
    for folder in sorted(base.iterdir()):
        if not folder.is_dir(): continue
        lab=LABELS.get(folder.name.lower().replace(' ', '_'))
        if lab is None: continue
        items.extend((str(p),lab) for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix.lower() in EXTS)
    if not items: raise RuntimeError(f'No videos found in {base}. Expected normal/ and anomalous/ folders')
    return items

def letterbox(rgb,w,h):
    ih,iw=rgb.shape[:2]
    if ih<1 or iw<1: raise ValueError('Invalid frame dimensions')
    scale=min(w/iw,h/ih)
    nw,nh=min(w,max(1,round(iw*scale))),min(h,max(1,round(ih*scale)))
    resized=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    # Padding is black in RGB; padding is applied before dataset/model normalization.
    out=np.zeros((h,w,3),dtype=np.uint8)
    x=(w-nw)//2;y=(h-nh)//2;out[y:y+nh,x:x+nw]=resized
    return out

def decode_one(path, frames, width, height):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f'Cannot decode {path}')
    try:
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count<=0:
            all_frames=[]
            while True:
                ret,f=cap.read()
                if not ret:break
                all_frames.append(f)
            if not all_frames: raise RuntimeError(f'Empty video {path}')
            inds=np.linspace(0,len(all_frames)-1,frames).round().astype(int)
            return np.stack([letterbox(cv2.cvtColor(all_frames[i],cv2.COLOR_BGR2RGB),width,height) for i in inds])
        inds=np.linspace(0,count-1,frames).round().astype(int)
        selected=[]; last=None
        # Sequential decoding avoids repeated random seeks on compressed videos.
        cursor=0
        for target in inds:
            while cursor<=int(target):
                ret,f=cap.read()
                if not ret:break
                last=f;cursor+=1
            if last is None:raise RuntimeError(f'Bad video {path}')
            selected.append(letterbox(cv2.cvtColor(last,cv2.COLOR_BGR2RGB),width,height))
        return np.stack(selected)
    finally:cap.release()

class CachedVideos(Dataset):
    def __init__(self,items,cfg,cache_dir,split,ram=True):
        self.items=items;self.split=split
        self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.path=self.cache_dir/f"{split}_{cfg['frames']}f_{cfg['width']}x{cfg['height']}_letterbox_v1.npy"
        self.meta=self.path.with_suffix('.json')
        signature=[str(Path(a).resolve())+'|'+str(b)+'|'+str(Path(a).stat().st_size)+'|'+str(Path(a).stat().st_mtime_ns) for a,b in items]
        reuse=self.path.exists() and self.meta.exists()
        if reuse:
            try:reuse=json.loads(self.meta.read_text()).get('signature')==signature
            except Exception:reuse=False
        if not reuse:
            tmp=self.path.with_suffix('.tmp.npy')
            mm=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(items),cfg['frames'],cfg['height'],cfg['width'],3))
            print(f'Caching {split}: {len(items)} clips, {cfg["frames"]} frames, {cfg["width"]}x{cfg["height"]}',flush=True)
            for i,(path,_) in enumerate(items):
                mm[i]=decode_one(path,cfg['frames'],cfg['width'],cfg['height'])
                if (i+1)%100==0:print(f'  cached {i+1}/{len(items)}',flush=True)
            mm.flush();del mm;os.replace(tmp,self.path)
            self.meta.write_text(json.dumps({'signature':signature,'cfg':{k:cfg[k] for k in ('frames','width','height')}}))
        # Memory mapping keeps total RAM bounded, OS page cache still accelerates repeated epochs.
        self.arr=np.load(self.path,mmap_mode='r')
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        # Copy is intentional: augmentation must not modify the shared cached memmap.
        x=torch.from_numpy(np.array(self.arr[i],copy=True)).permute(0,3,1,2)
        return x,int(self.items[i][1])


class VideoMAEHFClassifier(nn.Module):
    """HF checkpoint is K400-finetuned, with only its final classifier replaced."""
    def __init__(self, backbone):
        super().__init__(); self.model=backbone
    def forward(self,x):
        z=self.model(pixel_values=x.permute(0,2,1,3,4).contiguous())
        return z.logits



def build_model(name,source_dir=None):
    from transformers import VideoMAEForVideoClassification
    source = {
        'videomae_v1_s': 'MCG-NJU/videomae-small-finetuned-kinetics',
        'videomae_v1_b': 'MCG-NJU/videomae-base-finetuned-kinetics',
    }[name]
    model, info = VideoMAEForVideoClassification.from_pretrained(
        source, output_loading_info=True)
    missing = info.get('missing_keys', [])
    unexpected = info.get('unexpected_keys', [])
    if missing:
        raise RuntimeError(f'{name}: missing pretrained checkpoint weights: {missing[:15]}')
    # Unexpected keys can mean an incorrect checkpoint; do not silently proceed.
    if unexpected:
        raise RuntimeError(f'{name}: unexpected pretrained checkpoint weights: {unexpected[:15]}')
    if not isinstance(model.classifier, nn.Linear):
        raise RuntimeError(f'{name}: unexpected VideoMAE classification head type: {type(model.classifier)}')
    if model.classifier.out_features != 400:
        raise RuntimeError(f'{name}: expected Kinetics-400 pretrained head, got {model.classifier.out_features} classes')
    if getattr(model.config, 'num_frames', 16) != 16:
        raise RuntimeError(f'{name}: checkpoint does not use 16 frames')
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        except (TypeError, ValueError, AttributeError):
            model.gradient_checkpointing_enable()
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError(f'{name}: unexpectedly frozen backbone; full fine-tuning required')
    print(f'{name}: full K400-pretrained backbone loaded, new binary head, all parameters trainable',flush=True)
    return VideoMAEHFClassifier(model), f'{source}; K400 fine-tuned; full SCAD fine-tuning'

def forward_batch(model,x,y,cfg,train=False):
    # Input x is uint8 [B,T,C,H,W]. Video transforms on GPU, using normalizations of the pretrained model.
    x=x.to('cuda',non_blocking=True).float().div_(255.0)
    if train:
        mask=(torch.rand(x.size(0),device=x.device)<.5)
        x[mask]=x[mask].flip(-1)
    mean=torch.tensor(cfg['mean'],device=x.device).view(1,1,3,1,1)
    std=torch.tensor(cfg['std'],device=x.device).view(1,1,3,1,1)
    x=((x-mean)/std).permute(0,2,1,3,4).contiguous()
    logits=model(x)
    if logits.ndim!=2 or logits.shape[1]!=2:
        raise RuntimeError(f'Expected [batch,2] logits, got {tuple(logits.shape)}')
    return logits,y.to('cuda',non_blocking=True)

def eval_model(model,loader,cfg,amp_dtype):
    model.eval(); probs=[];labels=[];losses=[]
    with torch.inference_mode():
        for x,y in loader:
            with torch.autocast('cuda',dtype=amp_dtype):out,yy=forward_batch(model,x,y,cfg)
            losses.append((F.cross_entropy(out.float(),yy,reduction='sum').item(),len(y)))
            probs.extend(out.float().softmax(-1)[:,1].cpu().tolist());labels.extend(y.tolist())
    pred=[int(p>=.5) for p in probs]
    d=dict(loss=sum(a for a,_ in losses)/sum(b for _,b in losses),accuracy=accuracy_score(labels,pred)*100,
           f1=f1_score(labels,pred,zero_division=0)*100,precision=precision_score(labels,pred,zero_division=0)*100,
           recall=recall_score(labels,pred,zero_division=0)*100,
           auc=roc_auc_score(labels,probs) if len(set(labels))==2 else None,
           confusion_matrix=confusion_matrix(labels,pred,labels=[0,1]).tolist())
    return d

def preflight_model(name, source_dir, verbose=True):
    """Verify pretrained model can generate valid binary logits before any data caching/training.

    Uses synthetic uint8 RGB video in the same shape as CachedVideos; does not inspect test data.
    """
    cfg = SPECS[name]
    seed_all(42)
    torch.cuda.empty_cache()
    model = None
    try:
        model, pretraining = build_model(name, source_dir)
        model = model.cuda().eval()
        input_batch = torch.zeros((1, cfg['frames'], 3, cfg['height'], cfg['width']), dtype=torch.uint8)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad(), torch.autocast('cuda', dtype=amp_dtype):
            logits, _ = forward_batch(model, input_batch, torch.tensor([0]), cfg, train=False)
        if not torch.is_tensor(logits):
            raise TypeError(f'Model returned {type(logits)}, expected tensor [B,2]')
        if logits.shape != (1,2):
            raise RuntimeError(f'Model returned {tuple(logits.shape)}, expected (1,2)')
        if not torch.isfinite(logits).all().item():
            raise RuntimeError('Model produced non-finite logits')
        if verbose:
            print(f'PREFLIGHT PASS {name}: output={tuple(logits.shape)} | {pretraining}',flush=True)
        return True
    except Exception as exc:
        if verbose:
            print(f'PREFLIGHT FAIL {name}: {exc}\n{traceback.format_exc()}',flush=True)
        return False
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

def choose_batch_size(model,cfg,args):
    """A quick backward-pass batch probe, with safe VRAM margin for AdamW states.

    Throughput on synthetic frames is indicative only. It avoids prefetched-data stalls
    and prioritizes feasible batch sizes while keeping effective batch size 32.
    """
    if not args.autobatch:
        return cfg['batch']
    target=32
    candidates=[1,2,4,8,16,32]
    amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    params=[p for p in model.parameters() if p.requires_grad]
    model.eval()  # No BN running-stat changes during probing
    scores=[]
    for bs in candidates:
        if bs<1:continue
        xb=yb=logits=out=yy=loss=None
        try:
            torch.cuda.empty_cache()
            xb=torch.zeros((bs,cfg['frames'],3,cfg['height'],cfg['width']),dtype=torch.uint8)
            yb=torch.zeros(bs,dtype=torch.long)
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start=time.monotonic()
            with torch.autocast('cuda',dtype=amp_dtype):
                logits,_=forward_batch(model,xb,yb,cfg,False)
                loss=F.cross_entropy(logits.float(),yb.cuda())
            loss.backward()
            if bs==1:
                backbone_grad=sum(p.grad is not None for n,p in model.named_parameters()
                                  if p.requires_grad and not any(k in n.lower() for k in
                                  ('classifier','head','new_fc','projection_head')))
                if backbone_grad<1:
                    raise RuntimeError('Backbone has no gradients: full fine-tuning not active')
                print(f'FULL FINETUNE GRADIENT CHECK: {backbone_grad} backbone tensors have gradients',flush=True)
            torch.cuda.synchronize()
            elapsed=time.monotonic()-start
            used=torch.cuda.max_memory_allocated()
            total=torch.cuda.get_device_properties(0).total_memory
            # Gradients, AdamW optimizer states and input/cache buffers also need VRAM.
            if used>total*.72:
                print(f'BATCH PROBE {bs}: {used/2**30:.1f}GiB peak exceeds safe margin',flush=True)
                break
            scores.append((bs,bs/max(elapsed,1e-6),used/2**30))
            print(f'BATCH PROBE {bs}: {used/2**30:.1f}GiB, {bs/max(elapsed,1e-6):.1f} clips/s',flush=True)
        except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            print(f'BATCH PROBE {bs}: unavailable ({str(exc)[:120]})',flush=True)
            break
        finally:
            del xb,yb,logits,out,yy,loss
            model.zero_grad(set_to_none=True)
            gc.collect();torch.cuda.empty_cache()
    if not scores:
        raise RuntimeError('Even batch size 1 failed the backward-pass probe')
    # Largest feasible candidate within 10% of the top measured throughput.
    best_speed=max(row[1] for row in scores)
    choices=[row for row in scores if row[1]>=best_speed*.9]
    return max(choices,key=lambda row:row[0])[0]

def train(name,root,args):
    cfg=dict(SPECS[name]);save=Path(args.out);save.mkdir(parents=True,exist_ok=True)
    prefix=f'scad_variant_{name}'
    seed_all(args.seed);torch.cuda.empty_cache()
    print(f'\n===== {name} | {cfg["width"]}x{cfg["height"]} | {cfg["frames"]} frames =====',flush=True)
    model,pretraining=build_model(name,args.sources);model.cuda()
    total_params=sum(p.numel() for p in model.parameters());trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{name}: parameters={total_params/1e6:.2f}M; trainable={trainable_params/1e6:.2f}M',flush=True)
    # Fail fast before decoding/caching hundreds of clips.
    xb = torch.zeros((1,cfg['frames'],3,cfg['height'],cfg['width']),dtype=torch.uint8)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        logits,_ = forward_batch(model,xb,torch.zeros(1,dtype=torch.long),cfg,False)
    if logits.shape != (1,2) or not torch.isfinite(logits).all().item():
        raise RuntimeError(f'{name}: invalid pre-training logits {getattr(logits,"shape",type(logits))}')
    print(f'{name}: classification forward shape OK: {tuple(logits.shape)}',flush=True)
    del xb, logits
    cfg['batch']=choose_batch_size(model,cfg,args)
    cfg['accum']=max(1,(32+cfg['batch']-1)//cfg['batch'])
    print(f'{name}: selected physical batch {cfg["batch"]}, accumulate {cfg["accum"]}; target effective batch about 32',flush=True)
    tr=CachedVideos(scan(root,'train'),cfg,args.cache,'train')
    va=CachedVideos(scan(root,'val'),cfg,args.cache,'val')
    def loader(ds,shuffle):
        kw=dict(batch_size=cfg['batch'],shuffle=shuffle,num_workers=args.workers,pin_memory=True,
                persistent_workers=args.workers>0,drop_last=False)
        if args.workers:kw['prefetch_factor']=2
        return DataLoader(ds,**kw)
    tl=loader(tr,True);vl=loader(va,False)
    params=[p for p in model.parameters() if p.requires_grad]
    if 'head_lr' in cfg:
        head=[p for p in model.head.parameters() if p.requires_grad]
        head_ids={id(p) for p in head}
        back=[p for p in params if id(p) not in head_ids]
        opt=torch.optim.AdamW([{'params':back,'lr':cfg['lr']},{'params':head,'lr':cfg['head_lr']}],weight_decay=cfg['wd'])
    else:
        opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['wd'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=cfg['lr']*.05)
    amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=amp_dtype==torch.float16)
    best_acc=-1;best_loss=float('inf');hist=[];t0=time.time()
    meta={'model':name,'pretraining':pretraining,'input_width':cfg['width'],'input_height':cfg['height'],
          'frames':cfg['frames'],'seed':args.seed,'epochs':args.epochs,'batch':cfg['batch'],
          'parameters':total_params,'trainable_parameters':trainable_params,
          'accumulate':cfg['accum'],'lr':cfg['lr'],'weight_decay':cfg['wd'],'train_samples':len(tr),'val_samples':len(va),
          'test_policy':'test once after final best checkpoint, no test-driven model selection',
          'checkpoint_source':'official MCG-NJU Kinetics-400 fine-tuned VideoMAE V1'}
    (save/f'{prefix}_config.json').write_text(json.dumps(meta,indent=2))
    for ep in range(args.epochs):
        model.train();opt.zero_grad(set_to_none=True)
        loss_sum=0.0;n=0;correct=0;start=time.time()
        for j,(x,y) in enumerate(tl):
            with torch.autocast('cuda',dtype=amp_dtype):
                out,yy=forward_batch(model,x,y,cfg,True)
                loss=F.cross_entropy(out.float(),yy)
            scaler.scale(loss/cfg['accum']).backward()
            if ((j+1)%cfg['accum']==0) or j+1==len(tl):
                scaler.unscale_(opt);nn.utils.clip_grad_norm_(params,5.)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            loss_sum+=loss.item()*len(y);n+=len(y);correct+=(out.argmax(1)==yy).sum().item()
        torch.cuda.synchronize()
        scheduler.step();val=eval_model(model,vl,cfg,amp_dtype)
        elapsed=time.time()-start
        log=dict(epoch=ep+1,train_loss=loss_sum/n,train_accuracy=100*correct/n,
                 val_accuracy=val['accuracy'],val_loss=val['loss'],val_f1=val['f1'],
                 val_auc=val['auc'],epoch_seconds=elapsed,train_clips_per_sec=n/elapsed)
        hist.append(log);print(json.dumps(log),flush=True)
        if val['accuracy']>best_acc or (val['accuracy']==best_acc and val['loss']<best_loss):
            best_acc=val['accuracy'];best_loss=val['loss']
            torch.save({'model':model.state_dict(),'epoch':ep+1,'validation':val,'configuration':meta},save/f'{prefix}_best.pt')
    (save/f'{prefix}_epochs.json').write_text(json.dumps(hist,indent=2))
    # Load checkpoint selected exclusively on validation. Now, and only now, read held-out test.
    checkpoint=torch.load(save/f'{prefix}_best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model']);model.cuda()
    del tl,vl,tr,va;gc.collect()
    test=CachedVideos(scan(root,'test'),cfg,args.cache,'test')
    te=loader(test,False)
    test_metrics=eval_model(model,te,cfg,amp_dtype)
    result={**meta,'best_epoch':checkpoint['epoch'],'best_val_accuracy':best_acc,
            'test_accuracy':test_metrics['accuracy'],'test_f1':test_metrics['f1'],'test_auc':test_metrics['auc'],
            'test_precision':test_metrics['precision'],'test_recall':test_metrics['recall'],
            'confusion_matrix':test_metrics['confusion_matrix'],'wall_seconds':time.time()-t0}
    (save/f'{prefix}_result.json').write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result),flush=True)
    del te,test,model;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description='SCAD VideoMAE V1 Small/Base fine-tuning, sequential L40S training')
    p.add_argument('--data',default=str(here/'accident'))
    p.add_argument('--out',default=str(here),help='flat checkpoint/CSV outputs; default: beside Python file')
    p.add_argument('--cache',default=str(Path.home()/'.cache'/'scad_variants'/'frames'))
    p.add_argument('--sources',default=str(Path.home()/'.cache'/'scad_variants'/'sources'))
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--autobatch',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--rerun',action='store_true',help='Repeat and overwrite successful runs; default skips existing result files')
    p.add_argument('--models',nargs='+',choices=list(SPECS),default=None,
                   help='Only run these named architectures')
    a=p.parse_args()
    a.out=Path(a.out);a.out.mkdir(parents=True,exist_ok=True)
    if not Path(a.data,'train').is_dir() or not Path(a.data,'val').is_dir() or not Path(a.data,'test').is_dir():
        raise SystemExit(f'Dataset path does not have train/val/test: {a.data}')
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    if not torch.cuda.is_available():raise SystemExit('This run requires CUDA GPU')
    os.environ.setdefault('OMP_NUM_THREADS','2')
    torch.set_num_threads(min(2,os.cpu_count() or 2))
    print(f'CUDA device: {torch.cuda.get_device_name()} | source: {a.data} | seed={a.seed} | epochs={a.epochs}',flush=True)
    desired=a.models if a.models else list(SPECS)
    results=[];failures=[]
    for name in desired:
        actual=name
        prior=a.out/f'scad_variant_{name}_result.json'
        if prior.exists() and not a.rerun and not a.preflight_only:
            print(f'SKIP {name}: existing successful result {prior}',flush=True)
            continue
        if a.preflight_only:
            # Check before decoding; preflight model built once per selected variant
            ok=preflight_model(name,a.sources)
            if not ok:failures.append({'model':actual,'error':'preflight failed'})
            continue
        try:
            r=train(name,a.data,a)
            results.append(r)
        except Exception as exc:
            print(f'FAILED {name}: {exc}\n{traceback.format_exc()}',file=sys.stderr,flush=True)
            failures.append({'model':name,'error':str(exc),'traceback':traceback.format_exc()})
        gc.collect();torch.cuda.empty_cache()
        # Merge completed results from ALL invocations, preserving earlier batches.
        completed=[]
        for result_path in sorted(a.out.glob('scad_variant_*_result.json')):
            try:completed.append(json.loads(result_path.read_text()))
            except Exception as exc:print(f'WARNING: cannot read {result_path}: {exc}',flush=True)
        if completed:
            keys=['model','input_width','input_height','frames','batch','accumulate','best_epoch','best_val_accuracy','test_accuracy','test_f1','test_auc','test_precision','test_recall','wall_seconds']
            with open(a.out/'scad_variant_summary.csv','w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(completed)
        if failures:(a.out/'scad_variant_failures.json').write_text(json.dumps(failures,indent=2))
    if a.preflight_only:
        if failures:sys.exit(2)
        print('All selected model forward checks passed.',flush=True)
        return
    print('COMPLETE',len(results),'trained and evaluated;',len(failures),'failed.',flush=True)
    if failures:sys.exit(2)

if __name__=='__main__':main()
