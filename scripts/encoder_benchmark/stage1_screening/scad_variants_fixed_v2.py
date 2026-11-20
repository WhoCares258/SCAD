#!/usr/bin/env python3
"""SCAD architecture extension, L40S, single-file PySlowFast and UniFormerV2 compatibility fixes.

    cd /home/user/22104053
    source venv/bin/activate
    python -u scad_variants_l40s.py --models r2plus1d_18 slowfast_r50 timesformer

Data: ./accident/{train,val,test}/{normal,anomalous}/. Saves checkpoints and
metrics beside this file; model sources/frame caches under ~/.cache/scad_next/.

The next ten architectures have distinct official/model-zoo sources.
Adapter failures are reported explicitly: no random initialization or architecture substitutions.
All runs: uniform full 4-second sampling, RGB, resize + letterbox, 10 epochs,
seed 42, validation-selected checkpoint, independent test evaluated afterwards.
Different pretraining/input resolutions prevent an architecture-only attribution.
"""
import argparse, csv, gc, json, os, random, subprocess, sys, time, traceback, urllib.request, importlib, importlib.util, hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

SPECS = {
    'r2plus1d_18':   dict(width=112,height=112,frames=16,batch=8,lr=1e-4,wd=1e-4,mean=(.43216,.394666,.37645),std=(.22803,.22145,.216989),mode='bcthw'),
    'slowfast_r50':  dict(width=224,height=224,frames=32,batch=4,lr=1e-4,wd=1e-4,mean=(.45,)*3,std=(.225,)*3,mode='slowfast'),
    'timesformer':   dict(width=224,height=224,frames=8,batch=2,lr=5e-5,wd=.05,mean=(.45,)*3,std=(.225,)*3,mode='btchw'),
    'mvitv2_b':      dict(width=224,height=224,frames=32,batch=2,lr=5e-5,wd=.05,mean=(.45,)*3,std=(.225,)*3,mode='pyslowfast'),
    'videomaev2_s':  dict(width=224,height=224,frames=16,batch=4,lr=5e-5,wd=.05,mean=(.5,)*3,std=(.5,)*3,mode='bcthw'),
    'videomaev2_b_320':dict(width=320,height=320,frames=16,batch=2,lr=5e-5,wd=.05,mean=(.5,)*3,std=(.5,)*3,mode='bcthw'),
    'uniformerv2_b': dict(width=224,height=224,frames=8,batch=2,lr=5e-5,wd=.05,mean=(.48145,.45783,.40821),std=(.26863,.2613,.27578),mode='pyslowfast'),
    'internvideo2_l':dict(width=224,height=224,frames=8,batch=2,lr=2e-5,wd=.05,mean=(.485,.456,.406),std=(.229,.224,.225),mode='bcthw'),
    'videoprism_b': dict(width=288,height=288,frames=16,batch=2,lr=1e-5,head_lr=1e-4,wd=.05,mean=(0.,)*3,std=(1.,)*3,mode='bthwc'),
    'vjepa21_l':    dict(width=384,height=384,frames=16,batch=1,lr=1e-5,head_lr=1e-4,wd=.05,mean=(.485,.456,.406),std=(.229,.224,.225),mode='bcthw'),
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

def collect_state(ckpt):
    if isinstance(ckpt,dict):
        for k in ('model','module','state_dict','model_state_dict','model_state','state_dict_ema'):
            if isinstance(ckpt.get(k),dict):ckpt=ckpt[k];break
    if not isinstance(ckpt,dict):raise TypeError('Checkpoint has no recognized tensor state dictionary')
    return {k.removeprefix('module.'):v for k,v in ckpt.items() if torch.is_tensor(v)}

def load_matching(model,checkpoint,tag,min_coverage=.65):
    raw=collect_state(checkpoint);target=model.state_dict();good={}
    for k,v in raw.items():
        candidates=[k,k.removeprefix('backbone.'),k.removeprefix('encoder.'),
                    'backbone.'+k,'encoder.'+k]
        for name in candidates:
            if name in target and target[name].shape==v.shape:
                good[name]=v;break
    weighted=sum(v.numel() for v in good.values())/max(1,sum(v.numel() for v in target.values()))
    print(f'{tag} checkpoint match: {len(good)} tensors, {weighted*100:.1f}% of model parameters',flush=True)
    if weighted<min_coverage:raise RuntimeError(f'{tag}: insufficient pretrained weights matched: {weighted:.1%} (required {min_coverage:.0%}); will not train random model')
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

class C3DConv(nn.Module):
    """MMAction2-compatible naming for its released C3D checkpoint."""
    def __init__(self, inp, out):
        super().__init__()
        self.conv=nn.Conv3d(inp,out,kernel_size=3,padding=1)
        self.activate=nn.ReLU(inplace=True)
    def forward(self,x): return self.activate(self.conv(x))

class C3DClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a=C3DConv(3,64); self.pool1=nn.MaxPool3d((1,2,2),(1,2,2))
        self.conv2a=C3DConv(64,128); self.pool2=nn.MaxPool3d(2,2)
        self.conv3a=C3DConv(128,256); self.conv3b=C3DConv(256,256); self.pool3=nn.MaxPool3d(2,2)
        self.conv4a=C3DConv(256,512); self.conv4b=C3DConv(512,512); self.pool4=nn.MaxPool3d(2,2)
        self.conv5a=C3DConv(512,512); self.conv5b=C3DConv(512,512)
        self.pool5=nn.MaxPool3d((2,2,2),(2,2,2),padding=(0,1,1))
        self.fc6=nn.Linear(8192,4096); self.fc7=nn.Linear(4096,4096)
        self.relu=nn.ReLU(); self.dropout=nn.Dropout(.5); self.fc_cls=nn.Linear(4096,2)
    def forward(self,x):
        x=self.pool1(self.conv1a(x)); x=self.pool2(self.conv2a(x))
        x=self.pool3(self.conv3b(self.conv3a(x))); x=self.pool4(self.conv4b(self.conv4a(x)))
        x=self.pool5(self.conv5b(self.conv5a(x))).flatten(1)
        x=self.dropout(self.relu(self.fc6(x)))
        x=self.relu(self.fc7(x))
        return self.fc_cls(x)

class TrainableVideoEncoder(nn.Module):
    """Full end-to-end fine-tuning, no no_grad and encoder parameters trainable."""
    def __init__(self,encoder,dim=768,next_hf=False):
        super().__init__();self.encoder=encoder;self.next_hf=next_hf
        for p in encoder.parameters():p.requires_grad_(True)
        self.head=nn.Sequential(nn.LayerNorm(dim),nn.Dropout(.2),nn.Linear(dim,2))
    def forward(self,x):
        if self.next_hf:
            z=self.encoder.extract_features(pixel_values=x)
        else:
            z=self.encoder(x)
        if isinstance(z,dict):z=z.get('last_hidden_state',next(iter(z.values())))
        if isinstance(z,(list,tuple)):z=z[-1]
        if z.ndim==3:z=z.mean(1)
        elif z.ndim==5:z=z.mean((2,3,4))
        if z.ndim!=2:raise RuntimeError(f'Video feature shape unsupported: {tuple(z.shape)}')
        return self.head(z.float())

def _load_scad_c3d():
    # Official MMAction2 Sports-1M -> UCF101 checkpoint; no MMCV needed.
    model=C3DClassifier()
    url=('https://download.openmmlab.com/mmaction/v1.0/recognition/c3d/'
         'c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb/'
         'c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200.pth')
    ckpt=torch.hub.load_state_dict_from_url(url,map_location='cpu')
    raw=ckpt.get('state_dict',ckpt)
    state={k.removeprefix('backbone.'):v for k,v in raw.items() if k.startswith('backbone.')}
    matched={k:v for k,v in state.items() if k in model.state_dict() and model.state_dict()[k].shape==v.shape}
    coverage=sum(x.numel() for x in matched.values())/sum(v.numel() for k,v in model.state_dict().items() if k!='fc_cls.weight' and k!='fc_cls.bias')
    print(f'C3D pretrained backbone coverage: {coverage:.1%}',flush=True)
    if coverage<.90:raise RuntimeError('C3D official checkpoint mismatch; refusing random-weight training')
    model.load_state_dict(matched,strict=False)
    return model,'MMAction2 C3D Sports-1M -> UCF101 (backbone fine-tuned)'

def _pytorchvideo_classifier(architecture):
    model=torch.hub.load('facebookresearch/pytorchvideo:main',architecture,pretrained=True,trust_repo=True)
    head=model.blocks[-1]
    # Keep the pretrained feature projection. Replace only pooling and classifier.
    if hasattr(head,'pool') and hasattr(head.pool,'pool'):
        head.pool.pool=nn.AdaptiveAvgPool3d((1,1,1))
    elif hasattr(head,'pool'):
        head.pool=nn.AdaptiveAvgPool3d((1,1,1))
    else:
        raise RuntimeError(f'{architecture}: unsupported head pooling {head}')
    if not isinstance(head.proj,nn.Linear):raise RuntimeError(f'{architecture}: unsupported head classifier {head.proj}')
    head.proj=nn.Linear(head.proj.in_features,2)
    return model,f'PyTorchVideo {architecture} Kinetics-400'

def _stub_unused_flash_imports():
    # Only supports no-fused InternVideo2 configuration. NEVER executes stubbed kernels.
    import types
    methods=[('flash_attn.modules.mlp','FusedMLP'),('flash_attn.ops.rms_norm','DropoutAddRMSNorm'),
             ('flash_attn.flash_attn_interface','flash_attn_varlen_qkvpacked_func'),
             ('flash_attn.bert_padding','unpad_input'),('flash_attn.bert_padding','pad_input'),
             ('flash_attn','flash_attn_func'),('flash_attn','flash_attn_varlen_func')]
    for path,attr in methods:
        try:
            getattr(__import__(path,fromlist=[attr]),attr)
            continue
        except (ImportError,ModuleNotFoundError,AttributeError):pass
        bits=path.split('.')
        for j in range(1,len(bits)+1):
            m='.'.join(bits[:j]);sys.modules.setdefault(m,types.ModuleType(m))
            if j>1:setattr(sys.modules['.'.join(bits[:j-1])],bits[j-1],sys.modules[m])
        def unavailable(*args,**kw):raise RuntimeError('Unused FlashAttention stub was actually executed')
        setattr(sys.modules[path],attr,unavailable)

    # Transformers 4.57+ calls importlib.util.find_spec('flash_attn') while
    # importing PreTrainedModel. A bare ModuleType stub has __spec__ = None,
    # which raises ValueError instead of returning unavailable. Give the
    # import-only placeholders valid ModuleSpecs; they are NEVER CUDA kernels.
    from importlib.machinery import ModuleSpec
    for module_name, module in tuple(sys.modules.items()):
        if module_name == 'flash_attn' or module_name.startswith('flash_attn.'):
            if getattr(module, '__spec__', None) is None:
                is_package = any(key.startswith(module_name + '.') for key in sys.modules)
                module.__spec__ = ModuleSpec(module_name, loader=None, is_package=is_package)
                module.__package__ = module_name if is_package else module_name.rpartition('.')[0]
                if is_package:
                    module.__path__ = []

def _install_internvideo_next_sdpa(enc):
    """Run pretrained InternVideo-Next self-attention with memory-efficient
    torch.nn.functional.scaled_dot_product_attention rather than the enormous
    explicit N x N attention allocation used by upstream non-fused mode.
    This changes attention kernel, not model parameters or attention equations.
    """
    import types

    def native_attn(self, x):
        b,n,c=x.shape
        qkv=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads)
        q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
        if self.qk_normalization:
            _,h,nn,d=q.shape
            q=self.q_norm(q.transpose(1,2).flatten(-2,-1)).view(b,nn,h,d).transpose(1,2)
            k=self.k_norm(k.transpose(1,2).flatten(-2,-1)).view(b,nn,h,d).transpose(1,2)
        z=F.scaled_dot_product_attention(
            q,k,v,dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,scale=self.scale)
        return self.proj_drop(self.proj(z.transpose(1,2).reshape(b,n,c)))

    replaced=0
    for module in enc.modules():
        if (type(module).__name__=='Attention' and
                hasattr(module,'_naive_attn') and
                hasattr(module,'qkv') and hasattr(module,'num_heads')):
            if getattr(module,'use_flash_attn',True):
                raise RuntimeError('InternVideo-Next fused attention still enabled; cannot safely patch')
            module._naive_attn=types.MethodType(native_attn,module)
            replaced+=1
    if replaced!=12:
        raise RuntimeError(f'Expected InternVideo-Next Base 12 attention blocks, got {replaced}')
    print(f'InternVideo-Next: installed native SDPA in {replaced} transformer blocks',flush=True)

class SlowFastAdapter(nn.Module):
    def __init__(self,m): super().__init__();self.model=m
    def forward(self,x):
        # 32 uniformly sampled SCAD frames, fast path 32, slow path 8.
        return self.model([x[:,:,::4].contiguous(),x])

class BTCHWClassifier(nn.Module):
    def __init__(self,m): super().__init__();self.model=m
    def forward(self,x):
        z=self.model(pixel_values=x.permute(0,2,1,3,4))
        return z.logits

class PySlowFastAdapter(nn.Module):
    def __init__(self,m):super().__init__();self.model=m
    def forward(self,x):return self.model([x])

class PrismAdapter(nn.Module):
    """Fully trainable 2024 VideoPrism-B encoder via independently verified PyTorch port."""
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        self.head=nn.Sequential(nn.LayerNorm(768),nn.Dropout(.2),nn.Linear(768,2))
    def forward(self,x):
        # [B,C,T,H,W] -> VideoPrism's [B,T,H,W,C]. Input kept in [0,1].
        z=self.encoder(x.permute(0,2,3,4,1).contiguous())
        if hasattr(z,'last_hidden_state'):z=z.last_hidden_state
        if isinstance(z,dict):z=z['last_hidden_state']
        if z.ndim!=3 or z.shape[-1]!=768:
            raise RuntimeError(f'Unexpected VideoPrism encoder features: {tuple(z.shape)}')
        return self.head(z.mean(1).float())

def _video_mae(name,source_dir):
    repo=install_source_repo('https://github.com/OpenGVLab/VideoMAEv2.git',
        Path(source_dir)/'VideoMAEv2','models')
    purge_vendor_models();sys.path.insert(0,str(repo))
    try:
        from models.modeling_finetune import vit_small_patch16_224,vit_base_patch16_224
        is_small=name=='videomaev2_s'
        size=320 if name=='videomaev2_b_320' else 224
        constructor=vit_small_patch16_224 if is_small else vit_base_patch16_224
        model=constructor(img_size=size,all_frames=16,num_classes=2)
    finally:sys.path.remove(str(repo))
    ckpt_name='vit_s_k710_dl_from_giant.pth' if is_small else 'vit_b_k710_dl_from_giant.pth'
    ckpt=hub_download('OpenGVLab/VideoMAE2','distill/'+ckpt_name)
    raw=collect_state(torch.load(ckpt,map_location='cpu',weights_only=False))
    if size!=224:
        # Only 224^2 has an official VideoMAE V2 checkpoint. Recompute positional
        # embeddings via timm/model constructor at 320, while loading all backbone
        # weights. No pretense that this is an official 320 checkpoint.
        # Learnable position tokens, if present, must be interpolated explicitly.
        target=model.state_dict()
        for key in list(raw):
            if 'pos_embed' in key and key in target and raw[key].shape!=target[key].shape:
                a,b=raw[key],target[key]
                if a.ndim==3 and b.ndim==3 and a.shape[-1]==b.shape[-1]:
                    old=a.shape[1];new=b.shape[1]
                    t=8 # VideoMAEv2 uses 16 frames with tubelet_size=2
                    oside=int((old/t)**.5);nside=int((new/t)**.5)
                    if oside*oside*t==old and nside*nside*t==new:
                        tok=a.reshape(1,t,oside,oside,a.shape[-1]).permute(0,1,4,2,3).reshape(t,a.shape[-1],oside,oside)
                        raw[key]=F.interpolate(tok.float(),size=(nside,nside),mode='bicubic',align_corners=False).to(a.dtype).reshape(1,t,a.shape[-1],nside,nside).permute(0,1,3,4,2).reshape(1,new,a.shape[-1])
                        print(f'Interpolated {key}: {old} -> {new} position tokens',flush=True)
                    else:raise RuntimeError(f'Unable to interpolate VideoMAE pos_embed {old} -> {new}')
                else:raise RuntimeError(f'Unexpected VideoMAE positional embedding shape for {key}')
    load_matching(model,raw,f'VideoMAE V2 {"S" if is_small else "B-320"}')
    return model, ('VideoMAE V2-S K710 distilled' if is_small else 'VideoMAE V2-B official 224 K710 weights; SCAD custom 320 position adaptation')

def _internvideo_large(source_dir):
    repo=install_source_repo('https://github.com/OpenGVLab/InternVideo.git',
        Path(source_dir)/'InternVideo','InternVideo2/single_modality/models')
    models_dir=repo/'InternVideo2'/'single_modality'/'models';init=models_dir/'__init__.py'
    if 'from .videomae import' in init.read_text():
        init.write_text('from .internvideo2 import internvideo2_large_patch14_224\n')
    _stub_unused_flash_imports()
    sd=repo/'InternVideo2'/'single_modality'
    purge_vendor_models();sys.path.insert(0,str(sd))
    try:
        from models.internvideo2 import internvideo2_large_patch14_224
        model=internvideo2_large_patch14_224(pretrained=False,num_classes=2,num_frames=8,
            tubelet_size=1,use_flash_attn=False,use_fused_rmsnorm=False,
            use_fused_mlp=False,drop_path_rate=0.)
    finally:sys.path.remove(str(sd))
    ckpt=hub_download('OpenGVLab/InternVideo2_distillation_models',
        'stage1/L14/L14_ft_k710_ft_k400_f8/pytorch_model.bin')
    load_matching(model,torch.load(ckpt,map_location='cpu',weights_only=False),'InternVideo2-L')
    return model,'InternVideo2 distilled L14, K710 -> K400'

def checkpoint_transformer_blocks(module, tag):
    """Use non-reentrant activation checkpointing for memory-heavy ViTs."""
    from torch.utils.checkpoint import checkpoint
    blocks=getattr(module,'blocks',None)
    if blocks is None:
        print(f'{tag}: no encoder.blocks found; proceeding without block checkpointing',flush=True)
        return
    for block in blocks:
        original=block.forward
        def new_forward(*args,_original=original,_block=block,**kwargs):
            if _block.training and torch.is_grad_enabled():
                return checkpoint(_original,*args,use_reentrant=False,**kwargs)
            return _original(*args,**kwargs)
        block.forward=new_forward
    print(f'{tag}: non-reentrant activation checkpointing enabled for {len(blocks)} blocks',flush=True)

def _vjepa_large():
    enc,pred=torch.hub.load('facebookresearch/vjepa2:main','vjepa2_1_vit_large_384',
                            pretrained=False,num_frames=16,trust_repo=True)
    del pred
    url='https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'
    ckpt=torch.hub.load_state_dict_from_url(url,map_location='cpu')
    if 'ema_encoder' not in ckpt:raise RuntimeError('Missing official V-JEPA 2.1-L ema_encoder')
    raw={k.removeprefix('module.').removeprefix('backbone.'):v for k,v in ckpt['ema_encoder'].items()}
    missing,extra=enc.load_state_dict(raw,strict=False)
    print(f'V-JEPA-L official checkpoint: {len(missing)} missing, {len(extra)} unexpected',flush=True)
    if missing or extra:raise RuntimeError('V-JEPA 2.1-L weight mismatch')
    checkpoint_transformer_blocks(enc,'V-JEPA 2.1-L')
    return TrainableVideoEncoder(enc,1024),'Official V-JEPA 2.1-L EMA, 300M encoder, full fine-tuning'

def purge_pyslowfast_modules():
    # MViTv2-B and UniFormerV2 both vendor a `slowfast` namespace.
    # Prevent models/configuration from one repository leaking into the other.
    for key in list(sys.modules):
        if key == 'slowfast' or key.startswith('slowfast.'):
            del sys.modules[key]

def _ensure_pyslowfast_single_gpu_dependencies(source_dir):
    """Resolve PySlowFast's dependency on newer PyTorchVideo APIs without pip reinstall.

    We extend the installed package's search paths to the OFFICIAL GitHub code,
    preserving the already working environment and other models. This is for
    a single GPU only; it is not a substitute for distributed installation.
    """
    import pytorchvideo
    import pytorchvideo.layers
    from pytorchvideo.layers import distributed as old_distributed
    required = ('cat_all_gather', 'get_local_process_group', 'get_local_rank',
                'get_local_size', 'get_world_size', 'init_distributed_training')
    missing = [name for name in required if not hasattr(old_distributed, name)]
    if missing:
        repo = install_source_repo('https://github.com/facebookresearch/pytorchvideo.git',
                                   Path(source_dir) / 'pytorchvideo_official')
        lib = repo / 'pytorchvideo'
        if not (lib / 'layers' / 'distributed.py').is_file():
            raise RuntimeError('Official PyTorchVideo distributed.py was not cloned')
        # Make missing public subpackages such as pytorchvideo.losses available.
        for pkg, location in ((pytorchvideo, lib),
                              (pytorchvideo.layers, lib / 'layers')):
            loc = str(location)
            if loc not in pkg.__path__:
                pkg.__path__.insert(0, loc)
        sys.modules.pop('pytorchvideo.layers.distributed', None)
        patched = importlib.import_module('pytorchvideo.layers.distributed')
        still = [name for name in required if not hasattr(patched, name)]
        if still:
            raise RuntimeError(f'Official PyTorchVideo still lacks PySlowFast APIs: {still}')
        print(f'PySlowFast compatibility: official PyTorchVideo helpers loaded: {missing}', flush=True)

    # ROIAlign is needed only by detection architectures, not MViTv2 classification.
    # A dummy class guards against accidentally using a detection model.
    try:
        import detectron2.layers
    except ImportError:
        import types
        package = types.ModuleType('detectron2')
        package.__path__ = []
        package.__spec__ = importlib.util.spec_from_loader('detectron2', loader=None)
        layers = types.ModuleType('detectron2.layers')
        layers.__spec__ = importlib.util.spec_from_loader('detectron2.layers', loader=None)
        class UnsupportedDetectionROIAlign(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError('ROIAlign detection requires detectron2; SCAD uses classification only')
        layers.ROIAlign = UnsupportedDetectionROIAlign
        package.layers = layers
        sys.modules['detectron2'] = package
        sys.modules['detectron2.layers'] = layers
        print('PySlowFast compatibility: unused detectron2 ROIAlign import guarded (classification only)', flush=True)


def _mvitv2_base(source_dir):
    # The PyTorchVideo torchhub mvit_base_32x3 model is MViT V1, NOT V2.
    # Build the actual V2 model using the official PySlowFast config and weights.
    repo=install_source_repo('https://github.com/facebookresearch/SlowFast.git',Path(source_dir)/'SlowFast')
    _ensure_pyslowfast_single_gpu_dependencies(source_dir)
    try:
        import psutil  # imported unconditionally by PySlowFast utility modules
    except ImportError as e:
        raise RuntimeError('PySlowFast requires psutil: python -m pip install psutil') from e
    purge_pyslowfast_modules();sys.path.insert(0,str(repo))
    try:
        from slowfast.config.defaults import get_cfg
        from slowfast.models import build_model
        # Explicit official K400 B-32x3 config: never use MViTv1 or image-pretrained weights.
        paths=list(repo.glob('configs/Kinetics/MVITv2_B_32x3.y*ml'))
        if len(paths)!=1:
            raise RuntimeError('Expected official K400 config configs/Kinetics/MVITv2_B_32x3.yaml')
        cfg=get_cfg();cfg.merge_from_file(str(paths[0]));cfg.NUM_GPUS=1;cfg.MODEL.NUM_CLASSES=400
        model=build_model(cfg)
        url='https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth'
        ckpt=torch.hub.load_state_dict_from_url(url,map_location='cpu')
        load_matching(model,ckpt,'MViTv2-B')
        # Replace the K400 head only after checking pretrained backbone coverage.
        replaced=False
        for path in ['head.projection','head.proj']:
            a,b=path.split('.')
            if hasattr(model,a) and isinstance(getattr(getattr(model,a),b,None),nn.Linear):
                old=getattr(getattr(model,a),b)
                setattr(getattr(model,a),b,nn.Linear(old.in_features,2));replaced=True;break
        if not replaced:raise RuntimeError('MViTv2-B PySlowFast classification head layout unsupported')
        # Classification loss needs logits, not softmax-normalized probabilities.
        if hasattr(model.head, 'act'):
            model.head.act=nn.Identity()
        return PySlowFastAdapter(model),'Official MViTv2-B 32x3 K400 PySlowFast'
    finally:sys.path.remove(str(repo))

def _uniformerv2(source_dir):
    repo=install_source_repo('https://github.com/OpenGVLab/UniFormerV2.git',Path(source_dir)/'UniFormerV2')
    purge_pyslowfast_modules();sys.path.insert(0,str(repo))
    try:
        from slowfast.config.defaults import get_cfg
        from slowfast.models import build_model
        config=repo/'exp/k400/k400+k710_b16_f8x224/config.yaml'
        if not config.exists():raise RuntimeError(f'Missing {config}')
        cfg=get_cfg();cfg.merge_from_file(str(config));cfg.NUM_GPUS=1;cfg.MODEL.NUM_CLASSES=400
        cfg.UNIFORMERV2.PRETRAIN=''
        # Official class always passes pretrained=True to the CLIP initializer.
        # Suppress that LOCAL hardcoded CLIP load; load COMPLETE K400/K710
        # UniFormerV2 parameters below. Guard against silent random training.
        from slowfast.models import uniformerv2_model as uv2_arch
        original_constructor=uv2_arch.uniformerv2_b16
        def construct_without_local_clip(*args, **kwargs):
            kwargs['pretrained']=False
            return original_constructor(*args, **kwargs)
        uv2_arch.uniformerv2_b16=construct_without_local_clip
        try:
            model=build_model(cfg)
        finally:
            uv2_arch.uniformerv2_b16=original_constructor
        print('UniFormerV2: bypassed authors-only local CLIP path; loading official K400 video weights',flush=True)
        # Original UniFormerV2 OSS URL returns HTTP 404. Official author-hosted
        # HF repo is gated; this openly downloadable community MIRROR carries
        # the same K400+K710 B/16 8x224 filename. Its provenance is third-party:
        # verify full checkpoint coverage before interpreting benchmark results.
        mirror_repo='Andy1621/uniformerv2'
        mirror_file='k400+k710_uniformerv2_b16_8x224.pyth'
        print(f'UniFormerV2: downloading K400+K710 mirrored checkpoint {mirror_repo}/{mirror_file}',flush=True)
        ckpt_file=hub_download(mirror_repo,mirror_file)
        checkpoint=torch.load(ckpt_file,map_location='cpu',weights_only=True)
        load_matching(model,checkpoint,'UniFormerV2-B [community mirror]',min_coverage=.95)
        if not isinstance(model.backbone.transformer.proj[-1],nn.Linear):
            raise RuntimeError('UniFormerV2 classification head layout differs from published implementation')
        old=model.backbone.transformer.proj[-1]
        model.backbone.transformer.proj[-1]=nn.Linear(old.in_features,2)
        return PySlowFastAdapter(model),'UniFormerV2-B K710 -> K400 (Andy1621 HF mirror of author-published model; exact file provenance not independently verified)'
    finally:sys.path.remove(str(repo))

def _videoprism_base():
    # Google officially distributes JAX/Flax weights; this is an explicitly
    # attributed, numerically compared third-party PyTorch port of those weights.
    # No JAX stack added to the current SCAD virtualenv.
    model=torch.hub.load('Skovorp/torch_videoprism:main','videoprism_v1_base',
                         pretrained=True,trust_repo=True)
    return PrismAdapter(model),'Google VideoPrism-B public v1 Flax weights via Skovorp PyTorch port (verify independently)'

def build_model(name,source_dir):
    if name=='r2plus1d_18':
        from torchvision.models.video import r2plus1d_18,R2Plus1D_18_Weights
        model=r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        model.fc=nn.Linear(model.fc.in_features,2)
        return model,'Torchvision R(2+1)D-18 K400'
    if name=='slowfast_r50':
        model=torch.hub.load('facebookresearch/pytorchvideo:main','slowfast_r50',pretrained=True,trust_repo=True)
        head=model.blocks[-1]
        if not isinstance(head.proj,nn.Linear):raise RuntimeError('Unexpected SlowFast classifier')
        head.proj=nn.Linear(head.proj.in_features,2)
        return SlowFastAdapter(model),'PyTorchVideo SlowFast-R50 K400 8+32 frames'
    if name=='timesformer':
        from transformers import TimesformerForVideoClassification
        model=TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k400')
        old=model.classifier
        if not isinstance(old,nn.Linear):raise RuntimeError('Unexpected HF TimeSformer head')
        model.classifier=nn.Linear(old.in_features,2)
        return BTCHWClassifier(model),'Meta TimeSformer Base K400, 8 frames'
    if name=='mvitv2_b':return _mvitv2_base(source_dir)
    if name in ('videomaev2_s','videomaev2_b_320'):return _video_mae(name,source_dir)
    if name=='uniformerv2_b':return _uniformerv2(source_dir)
    if name=='internvideo2_l':return _internvideo_large(source_dir)
    if name=='videoprism_b':return _videoprism_base()
    if name=='vjepa21_l':return _vjepa_large()
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
          'test_policy':'test once after final best checkpoint, no test-driven model selection'}
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
    p=argparse.ArgumentParser(description='SCAD ten-configuration architecture extension, single file, sequential L40S training')
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
