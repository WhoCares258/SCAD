#!/usr/bin/env python3
"""SCAD modern-model RGB screening, seed 42.

Run from ~/scad_screen after activating ~/venv:
    python -u scad_remaining_models.py

Dataset defaults to the sibling folder ../accident, with
train/, val/, test/ and normal/, anomalous/ class subfolders.

Model-specific pretrained input shapes; all frames letterboxed without crop.
Preflight binary logits before caching/training. Select checkpoint by validation
accuracy (validation loss tie-breaker); test once per run after selection.

Only the four newer models run by default. X3D-S is left undisturbed.
Official research repos can change; adapters may require repair on the HPC.
"""
import argparse, csv, gc, json, os, random, subprocess, sys, time, traceback, urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

SPECS = {
    'x3d_s':       dict(width=480,height=360,frames=13,batch=4,accum=4,lr=1e-4,wd=1e-4,mean=(.45,)*3,std=(.225,)*3,mode='bcthw'),
    'mvitv2_s':    dict(width=224,height=224,frames=16,batch=2,accum=8,lr=5e-5,wd=.05,mean=(.45,)*3,std=(.225,)*3,mode='bcthw'),
    'videomaev2_b':dict(width=224,height=224,frames=16,batch=2,accum=8,lr=5e-5,wd=.05,mean=(.5,)*3,std=(.5,)*3,mode='bcthw'),
    'internvideo2_b':dict(width=224,height=224,frames=8,batch=2,accum=8,lr=5e-5,wd=.05,mean=(.485,.456,.406),std=(.229,.224,.225),mode='bcthw'),
    'vjepa21_b':   dict(width=384,height=384,frames=16,batch=1,accum=16,lr=1e-3,wd=1e-4,mean=(.485,.456,.406),std=(.229,.224,.225),mode='bcthw'),
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
        self.path=self.cache_dir/f"{split}_{cfg['frames']}f_{cfg['width']}x{cfg['height']}.npy"
        self.meta=self.path.with_suffix('.json')
        signature=[str(a)+'|'+str(b) for a,b in items]
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

def collect_state(ckpt):
    if isinstance(ckpt,dict):
        for k in ('model','module','state_dict','model_state_dict'):
            if isinstance(ckpt.get(k),dict):ckpt=ckpt[k];break
    return {k.removeprefix('module.').removeprefix('backbone.').removeprefix('encoder.'):v for k,v in ckpt.items() if torch.is_tensor(v)}

def load_matching(model,checkpoint,tag):
    raw=collect_state(checkpoint)
    target=model.state_dict()
    good={k:v for k,v in raw.items() if k in target and target[k].shape==v.shape}
    weighted=sum(v.numel() for v in good.values())/max(1,sum(v.numel() for v in target.values()))
    print(f'{tag} checkpoint match: {len(good)} tensors, {weighted*100:.1f}% of model parameters',flush=True)
    if weighted<.65:raise RuntimeError(f'{tag}: insufficient pretrained weights matched: {weighted:.1%}; will not train random model')
    model.load_state_dict(good,strict=False)

def install_source_repo(url,location,sparse_dir=None):
    location=Path(location)
    if location.exists():return location
    location.parent.mkdir(parents=True,exist_ok=True)
    if sparse_dir:
        subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',url,str(location)],check=True)
        subprocess.run(['git','-C',str(location),'sparse-checkout','set',sparse_dir],check=True)
    else:subprocess.run(['git','clone','--depth','1',url,str(location)],check=True)
    return location

def hub_download(repo_id,filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id,filename=filename)

class FrozenVideoEncoder(nn.Module):
    def __init__(self,encoder,feature_dim,expects_btchw=False):
        super().__init__();self.encoder=encoder
        self.expects_btchw=expects_btchw
        for p in self.encoder.parameters():p.requires_grad_(False)
        self.encoder.eval()
        self.head=nn.Sequential(nn.LayerNorm(feature_dim),nn.Dropout(.2),nn.Linear(feature_dim,2))
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def forward(self,x):
        with torch.no_grad():
            z=self.encoder(x.permute(0,2,1,3,4) if self.expects_btchw else x)
        if isinstance(z,dict):z=z.get('last_hidden_state',next(iter(z.values())))
        if isinstance(z,(tuple,list)):z=z[-1]
        if z.ndim==3:z=z.mean(dim=1)
        if z.ndim==5:z=z.mean(dim=(2,3,4))
        if z.ndim!=2:raise RuntimeError(f'Unexpected encoder shape {z.shape}')
        return self.head(z.float())

def purge_vendor_models():
    # The two original research codebases both use a top-level package named "models".
    # Ensure the second import cannot reuse modules cached from the first codebase.
    for key in list(sys.modules):
        if key == 'models' or key.startswith('models.'):
            del sys.modules[key]

def build_model(name,source_dir):
    if name=='x3d_s':
        model=torch.hub.load('facebookresearch/pytorchvideo:main','x3d_s',pretrained=True,trust_repo=True)
        head=model.blocks[-1]
        if not hasattr(head, 'pool') or not hasattr(head.pool, 'pool'):
            raise RuntimeError(f'X3D unexpected projected pooling head: {type(head)}')
        head.pool.pool = nn.AdaptiveAvgPool3d((1,1,1))
        if not isinstance(head.proj,nn.Linear):raise RuntimeError(f'Unexpected X3D head: {type(head.proj)}')
        head.proj=nn.Linear(head.proj.in_features,2)
        return model,'Kinetics-400 PyTorchVideo'
    if name=='mvitv2_s':
        from torchvision.models.video import mvit_v2_s,MViT_V2_S_Weights
        model=mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        assert isinstance(model.head[-1],nn.Linear), f'Unexpected MViT head {model.head}'
        model.head[-1]=nn.Linear(model.head[-1].in_features,2)
        return model,'Torchvision Kinetics-400'
    if name=='videomaev2_b':
        repo=install_source_repo('https://github.com/OpenGVLab/VideoMAEv2.git',Path(source_dir)/'VideoMAEv2','models')
        purge_vendor_models()
        sys.path.insert(0,str(repo))
        try:
            from models.modeling_finetune import vit_base_patch16_224
            model=vit_base_patch16_224(img_size=224,all_frames=16,num_classes=2)
        finally:
            sys.path.remove(str(repo))
        assert isinstance(model.head,nn.Linear) and model.head.out_features == 2, f'VideoMAE head: {model.head}'
        ckpt_path=hub_download('OpenGVLab/VideoMAE2','distill/vit_b_k710_dl_from_giant.pth')
        ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        load_matching(model,ckpt,'VideoMAE V2-B')
        return model,'VideoMAE2 ViT-B distilled K710'
    if name=='internvideo2_b':
        repo=install_source_repo('https://github.com/OpenGVLab/InternVideo.git',Path(source_dir)/'InternVideo','InternVideo2/single_modality/models')
        # InternVideo2 source imports optional fused CUDA operations, even when disabled.
        # Stub only their unused imports in this non-fused configuration.
        import types
        for mod_name,attr in [('flash_attn.modules.mlp','FusedMLP'),('flash_attn.ops.rms_norm','DropoutAddRMSNorm'),
                              ('flash_attn.flash_attn_interface','flash_attn_varlen_qkvpacked_func'),
                              ('flash_attn.bert_padding','unpad_input'),('flash_attn.bert_padding','pad_input')]:
            try:
                getattr(__import__(mod_name,fromlist=[attr]),attr)
                continue
            except (ImportError,ModuleNotFoundError,AttributeError):
                bits=mod_name.split('.')
                for j in range(1,len(bits)+1):
                    n='.'.join(bits[:j]);sys.modules.setdefault(n,types.ModuleType(n))
                if attr in ('pad_input','unpad_input','flash_attn_varlen_qkvpacked_func'):
                    def disabled_flash(*args, **kwargs):
                        raise RuntimeError('Fused flash-attn kernel disabled by SCAD adapter')
                    setattr(sys.modules[mod_name],attr,disabled_flash)
                else:
                    class DisabledFusedLayer(nn.Module):
                        def __init__(self,*args,**kwargs):super().__init__()
                        def forward(self,*args,**kwargs):
                            raise RuntimeError('Fused flash-attn layer disabled by SCAD adapter')
                    setattr(sys.modules[mod_name],attr,DisabledFusedLayer)
        sd=repo/'InternVideo2'/'single_modality'
        purge_vendor_models()
        sys.path.insert(0,str(sd))
        try:
            # Source package registers model via timm; non-fused path avoids flash-attn compilation.
            from models.internvideo2 import internvideo2_base_patch14_224
            model=internvideo2_base_patch14_224(pretrained=False,num_classes=2,num_frames=8,tubelet_size=1,
                use_flash_attn=False,use_fused_rmsnorm=False,use_fused_mlp=False,drop_path_rate=0.0,
                init_values=1e-5)
        finally:sys.path.remove(str(sd))
        assert isinstance(model.head,nn.Linear) and model.head.out_features == 2, f'InternVideo2 head: {model.head}'
        ckpt_path=hub_download('OpenGVLab/InternVideo2_distillation_models',
            'stage1/B14/B14_ft_k710_ft_k400_f8/pytorch_model.bin')
        ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        load_matching(model,ckpt,'InternVideo2-B')
        return model,'InternVideo2-B K710-to-K400 adapted'
    if name=='vjepa21_b':
        enc,pred=torch.hub.load('facebookresearch/vjepa2:main','vjepa2_1_vit_base_384',pretrained=False,num_frames=16,trust_repo=True)
        del pred
        url='https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt'
        ckpt=torch.hub.load_state_dict_from_url(url,map_location='cpu',progress=True)
        if 'ema_encoder' not in ckpt:raise RuntimeError('V-JEPA2.1 official checkpoint missing ema_encoder')
        raw={k.removeprefix('module.').removeprefix('backbone.'):v for k,v in ckpt['ema_encoder'].items()}
        missing,unexpected=enc.load_state_dict(raw,strict=False)
        print('V-JEPA2.1 encoder missing/unexpected:',len(missing),len(unexpected))
        if len(missing)>10:raise RuntimeError('V-JEPA2.1 encoder not loaded correctly')
        return FrozenVideoEncoder(enc,768),'Official V-JEPA 2.1 ViT-B frozen encoder + binary head'
    raise ValueError(name)

def forward_batch(model,x,y,cfg,train=False):
    # Input x is uint8 [B,T,C,H,W]. Video transforms on GPU, using normalizations of the pretrained model.
    x=x.to('cuda',non_blocking=True).float().div_(255.0)
    if train:
        mask=(torch.rand(x.size(0),device=x.device)<.5)
        x[mask]=x[mask].flip(-1)
    mean=torch.tensor(cfg['mean'],device=x.device).view(1,1,3,1,1)
    std=torch.tensor(cfg['std'],device=x.device).view(1,1,3,1,1)
    x=((x-mean)/std).permute(0,2,1,3,4).contiguous()
    return model(x),y.to('cuda',non_blocking=True)

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

def train(name,root,args):
    cfg=SPECS[name];save=Path(args.out)/name;save.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed);torch.cuda.empty_cache()
    print(f'\n===== {name} | {cfg["width"]}x{cfg["height"]} | {cfg["frames"]} frames =====',flush=True)
    model,pretraining=build_model(name,args.sources);model.cuda()
    # Fail fast before decoding/caching hundreds of clips.
    xb = torch.zeros((1,cfg['frames'],3,cfg['height'],cfg['width']),dtype=torch.uint8)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        logits,_ = forward_batch(model,xb,torch.zeros(1,dtype=torch.long),cfg,False)
    if logits.shape != (1,2) or not torch.isfinite(logits).all().item():
        raise RuntimeError(f'{name}: invalid pre-training logits {getattr(logits,"shape",type(logits))}')
    print(f'{name}: classification forward shape OK: {tuple(logits.shape)}',flush=True)
    del xb, logits
    tr=CachedVideos(scan(root,'train'),cfg,args.cache,'train')
    va=CachedVideos(scan(root,'val'),cfg,args.cache,'val')
    def loader(ds,shuffle):
        kw=dict(batch_size=cfg['batch'],shuffle=shuffle,num_workers=args.workers,pin_memory=True,
                persistent_workers=args.workers>0,drop_last=False)
        if args.workers:kw['prefetch_factor']=2
        return DataLoader(ds,**kw)
    tl=loader(tr,True);vl=loader(va,False)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['wd'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=cfg['lr']*.05)
    amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=amp_dtype==torch.float16)
    best_acc=-1;best_loss=float('inf');hist=[];t0=time.time()
    meta={'model':name,'pretraining':pretraining,'input_width':cfg['width'],'input_height':cfg['height'],
          'frames':cfg['frames'],'seed':args.seed,'epochs':args.epochs,'batch':cfg['batch'],
          'accumulate':cfg['accum'],'lr':cfg['lr'],'weight_decay':cfg['wd'],'train_samples':len(tr),'val_samples':len(va),
          'test_policy':'test once after final best checkpoint, no test-driven model selection'}
    (save/'config.json').write_text(json.dumps(meta,indent=2))
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
        scheduler.step();val=eval_model(model,vl,cfg,amp_dtype)
        elapsed=time.time()-start
        log=dict(epoch=ep+1,train_loss=loss_sum/n,train_accuracy=100*correct/n,
                 val_accuracy=val['accuracy'],val_loss=val['loss'],val_f1=val['f1'],
                 val_auc=val['auc'],epoch_seconds=elapsed,train_clips_per_sec=n/elapsed)
        hist.append(log);print(json.dumps(log),flush=True)
        if val['accuracy']>best_acc or (val['accuracy']==best_acc and val['loss']<best_loss):
            best_acc=val['accuracy'];best_loss=val['loss']
            torch.save({'model':model.state_dict(),'epoch':ep+1,'validation':val,'configuration':meta},save/'best.pt')
    (save/'epochs.json').write_text(json.dumps(hist,indent=2))
    # Load checkpoint selected exclusively on validation. Now, and only now, read held-out test.
    checkpoint=torch.load(save/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model']);model.cuda()
    del tl,vl,tr,va;gc.collect()
    test=CachedVideos(scan(root,'test'),cfg,args.cache,'test')
    te=loader(test,False)
    test_metrics=eval_model(model,te,cfg,amp_dtype)
    result={**meta,'best_epoch':checkpoint['epoch'],'best_val_accuracy':best_acc,
            'test_accuracy':test_metrics['accuracy'],'test_f1':test_metrics['f1'],'test_auc':test_metrics['auc'],
            'test_precision':test_metrics['precision'],'test_recall':test_metrics['recall'],
            'confusion_matrix':test_metrics['confusion_matrix'],'wall_seconds':time.time()-t0}
    (save/'result.json').write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result),flush=True)
    del te,test,model;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=str(Path(__file__).resolve().parent.parent/'22104053/accident'),help='SCAD dataset folder with train/ val/ test/; defaults to ../accident')
    p.add_argument('--out',default='scad_screen_results')
    p.add_argument('--cache',default='scad_frame_cache')
    p.add_argument('--sources',default='scad_model_sources')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--preflight-only',action='store_true',help='Check all selected model forward passes without loading SCAD or training')
    p.add_argument('--models',nargs='+',choices=list(SPECS),default=['mvitv2_s','videomaev2_b','internvideo2_b','vjepa21_b'])
    a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
    if a.preflight_only:
        ok = [preflight_model(name,a.sources) for name in a.models]
        if not all(ok):sys.exit(2)
        print('All selected models passed pretrained binary-head forward checks.',flush=True)
        return
    results=[];failures=[]
    for name in a.models:
        try:
            r=train(name,a.data,a);results.append(r)
        except Exception as e:
            print(f'FAILED {name}: {e}\n{traceback.format_exc()}',file=sys.stderr,flush=True)
            failures.append({'model':name,'error':str(e),'traceback':traceback.format_exc()})
            (Path(a.out)/f'{name}_failure.json').write_text(json.dumps(failures[-1],indent=2))
        gc.collect();torch.cuda.empty_cache()
        if results:
            keys=['model','input_width','input_height','frames','best_epoch','best_val_accuracy','test_accuracy','test_f1','test_auc','test_precision','test_recall','wall_seconds']
            with open(Path(a.out)/'summary.csv','w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(results)
    if failures:print(f'{len(failures)} model(s) failed. See {a.out}/*_failure.json',file=sys.stderr)
    if not results:sys.exit(2)

if __name__=='__main__':main()
