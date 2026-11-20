#!/usr/bin/env python3
"""SCAD next batch, L40S, standalone Python file.

    cd /home/user/22104053
    source venv/bin/activate
    python -u scad_next_seven_no_flash_v2.py --models c3d i3d_r50 x3d_s_original vjepa21_full

Data: ./accident/{train,val,test}/{normal,anomalous}/. Saves checkpoints and
metrics beside this file; model sources/frame caches under ~/.cache/scad_next/.

Also supports InternVideo2-B and InternVideo-Next-B (official code/checkpoints).
VideoMambaPro requires a separate compatible CUDA stack and manually obtained
pretrained weights, so that model fails clearly instead of training random weights.
All runs: uniform full 4-second sampling, RGB, resize + letterbox, 10 epochs,
seed 42, validation-selected checkpoint, independent test evaluated afterwards.
Different pretraining/input resolutions prevent an architecture-only attribution.
"""
import argparse, csv, gc, json, os, random, subprocess, sys, time, traceback, urllib.request, importlib.util, hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

SPECS = {
    'c3d':                dict(width=112,height=112,frames=16,batch=8,accum=4,lr=1e-4,wd=1e-4,mean=(.485,.456,.406),std=(.229,.224,.225),mode='bcthw'),
    'i3d_r50':            dict(width=224,height=224,frames=8,batch=8,accum=4,lr=1e-4,wd=1e-4,mean=(.45,)*3,std=(.225,)*3,mode='bcthw'),
    'x3d_s_original':     dict(width=160,height=160,frames=13,batch=8,accum=4,lr=1e-4,wd=1e-4,mean=(.45,)*3,std=(.225,)*3,mode='bcthw'),
    'vjepa21_full':       dict(width=384,height=384,frames=16,batch=2,accum=16,lr=1e-5,head_lr=1e-4,wd=.05,mean=(.485,.456,.406),std=(.229,.224,.225),mode='bcthw'),
    'internvideo2_b':     dict(width=224,height=224,frames=8,batch=4,accum=8,lr=5e-5,wd=.05,mean=(.485,.456,.406),std=(.229,.224,.225),mode='bcthw'),
    'internvideo_next_b': dict(width=224,height=224,frames=16,batch=2,accum=16,lr=1e-5,head_lr=1e-4,wd=.05,mean=(.485,.456,.406),std=(.229,.224,.225),mode='bcthw'),
    'videomambapro':      dict(width=224,height=224,frames=16,batch=2,accum=16,lr=5e-5,wd=.05,mean=(.485,.456,.406),std=(.229,.224,.225),mode='bcthw'),
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
        for k in ('model','module','state_dict','model_state_dict','state_dict_ema'):
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

def build_model(name,source_dir):
    if name=='c3d':return _load_scad_c3d()
    if name=='i3d_r50':return _pytorchvideo_classifier('i3d_r50')
    if name=='x3d_s_original':return _pytorchvideo_classifier('x3d_s')
    if name=='vjepa21_full':
        enc,pred=torch.hub.load('facebookresearch/vjepa2:main','vjepa2_1_vit_base_384',pretrained=False,num_frames=16,trust_repo=True)
        del pred
        ckpt=torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt',map_location='cpu')
        raw={k.removeprefix('module.').removeprefix('backbone.'):v for k,v in ckpt['ema_encoder'].items()}
        missing,extra=enc.load_state_dict(raw,strict=False)
        if missing or extra:raise RuntimeError(f'V-JEPA pretrained encoder mismatch: missing={len(missing)}, extra={len(extra)}')
        return TrainableVideoEncoder(enc,768),'V-JEPA 2.1 ViT-B official EMA; FULL trainable encoder'
    if name=='internvideo2_b':
        repo=install_source_repo('https://github.com/OpenGVLab/InternVideo.git',Path(source_dir)/'InternVideo','InternVideo2/single_modality/models')
        # Package __init__ imports many unrelated teachers with FlashAttention.
        # Use a minimal initializer in *local cache copy* only.
        models_dir=repo/'InternVideo2'/'single_modality'/'models'
        init=models_dir/'__init__.py'
        if not init.exists():raise RuntimeError('InternVideo2 repository missing expected models package')
        if 'from .videomae import' in init.read_text():
            init.write_text('from .internvideo2 import internvideo2_base_patch14_224\n')
        _stub_unused_flash_imports()
        sd=repo/'InternVideo2'/'single_modality'
        purge_vendor_models();sys.path.insert(0,str(sd))
        try:
            from models.internvideo2 import internvideo2_base_patch14_224
            model=internvideo2_base_patch14_224(pretrained=False,num_classes=2,num_frames=8,tubelet_size=1,
                use_flash_attn=False,use_fused_rmsnorm=False,use_fused_mlp=False,drop_path_rate=0.0)
        finally:sys.path.remove(str(sd))
        ckpt_path=hub_download('OpenGVLab/InternVideo2_distillation_models',
            'stage1/B14/B14_ft_k710_ft_k400_f8/pytorch_model.bin')
        load_matching(model,torch.load(ckpt_path,map_location='cpu',weights_only=False),'InternVideo2-B')
        return model,'InternVideo2-B distilled, K710 -> K400'
    if name=='internvideo_next_b':
        from transformers import AutoConfig, AutoModel
        # Official source has unconditional flash_attn imports even in non-fused mode.
        # These stubs must NEVER be used for computation. We force all fused paths off.
        _stub_unused_flash_imports()
        repo='revliter/internvideo_next_base_p14_res224_f16'
        hf_cfg=AutoConfig.from_pretrained(repo, trust_remote_code=True)
        cfg=hf_cfg.model_config
        cfg.update(use_flash_attn=False,use_fused_rmsnorm=False,use_fused_mlp=False,
                   use_checkpoint=True,checkpoint_num=12)
        print('InternVideo-Next: non-fused layers + native PyTorch SDPA; gradient checkpointing on',flush=True)
        enc,info=AutoModel.from_pretrained(repo,config=hf_cfg,trust_remote_code=True,
                                            output_loading_info=True)
        # Avoid silently running a mostly random model if fused/nonfused names differ.
        missing=set(info.get('missing_keys',[]))
        state=enc.state_dict()
        missing_count=sum(v.numel() for k,v in state.items() if k in missing)
        total_count=sum(v.numel() for v in state.values())
        coverage=1-missing_count/max(total_count,1)
        print(f'InternVideo-Next pretrained parameter coverage: {coverage:.1%} | missing keys: {len(missing)}',flush=True)
        if coverage<.90 or info.get('mismatched_keys'):
            raise RuntimeError('InternVideo-Next weight mismatch after non-fused configuration; not training partial random weights')
        if not hasattr(enc,'extract_features'):
            raise RuntimeError('InternVideo-Next checkpoint lacks expected extract_features API')
        _install_internvideo_next_sdpa(enc)
        return TrainableVideoEncoder(enc,768,next_hf=True),'InternVideo-Next Base HF pretrained; non-fused PyTorch SDPA; full fine-tuning'
    if name=='videomambapro':
        # Upstream provides a modified mamba-ssm fork, not a drop-in dependency.
        # Upstream weights are not downloadable from an automated URL.
        raise RuntimeError('VideoMambaPro requires its custom CUDA 11.8 / torch 2.1.1 stack and manually downloaded '
            'checkpoint from https://github.com/hotfinda/VideoMambaPro (Baidu link). '
            'Cannot run an authenticated pretrained comparison in this venv without those assets. '
            'No random-weight fallback is permitted.')
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
    prefix=f'scad_next_{name}'
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
    p=argparse.ArgumentParser(description='SCAD seven-model screening, single file, sequential L40S training')
    p.add_argument('--data',default=str(here/'accident'))
    p.add_argument('--out',default=str(here),help='flat checkpoint/CSV outputs; default: beside Python file')
    p.add_argument('--cache',default=str(Path.home()/'.cache'/'scad_next'/'frames'))
    p.add_argument('--sources',default=str(Path.home()/'.cache'/'scad_next'/'sources'))
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--autobatch',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--rerun',action='store_true',help='Repeat and overwrite successful runs; default skips existing result files')
    p.add_argument('--models',nargs='+',choices=list(SPECS),default=None,
                   help='Optional specific models (for reruns), instead of all seven candidates')
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
        prior=a.out/f'scad_next_{name}_result.json'
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
        for result_path in sorted(a.out.glob('scad_next_*_result.json')):
            try:completed.append(json.loads(result_path.read_text()))
            except Exception as exc:print(f'WARNING: cannot read {result_path}: {exc}',flush=True)
        if completed:
            keys=['model','input_width','input_height','frames','batch','accumulate','best_epoch','best_val_accuracy','test_accuracy','test_f1','test_auc','test_precision','test_recall','wall_seconds']
            with open(a.out/'scad_next_summary.csv','w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(completed)
        if failures:(a.out/'scad_next_failures.json').write_text(json.dumps(failures,indent=2))
    if a.preflight_only:
        if failures:sys.exit(2)
        print('All selected model forward checks passed.',flush=True)
        return
    print('COMPLETE',len(results),'trained and evaluated;',len(failures),'failed.',flush=True)
    if failures:sys.exit(2)

if __name__=='__main__':main()
