#!/usr/bin/env python3
"""SCAD 17-configuration NVIDIA NVDEC end-to-end latency profiler.

NVDEC decode and RGB output on GPU via PyNvVideoCodec SimpleDecoder,
DLPack/torch zero-copy, GPU letterbox + normalization, full fine-tuned
model inference, and final prediction. Cold-per-clip path starts immediately
before constructing a decoder from a video filename. No OpenCV decode fallback.

Requirements:
  pip install PyNvVideoCodec
  scad_17_pipeline_v2.py and scad_17_sources/ alongside this file
  All 17 existing, trusted fine-tuned SCAD .pt checkpoints discoverable by v2

Commands:
  python -u scad_17_nvdec_e2e.py --stage check
  python -u scad_17_nvdec_e2e.py --stage prepare --videos 20
  python -u scad_17_nvdec_e2e.py --stage profile 2>&1 | tee scad_nvdec17.log
  python -u scad_17_nvdec_e2e.py --stage single --single x3d_m

Interpretation: serial batch-1 whole-clip latency, not streaming steady-state
throughput; file-backed NVDEC frame sampling can seek to keyframes. No frame
cache, no CPU-transfer of image data, no overlap across clips. GPU transforms
approximate the original OpenCV letterbox geometry and use same normalization,
but pixel-level interpolation/color differences can change predictions; the
script is a LATENCY benchmark, not a replacement test accuracy evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

DEFAULT_HOME=Path('/home/user/22104053')


def parent_module():
    name='_scad17_discovery'
    if name in sys.modules: return sys.modules[name]
    src=Path(__file__).resolve().parent/'scad_17_pipeline_v2.py'
    if not src.is_file():
        raise FileNotFoundError(f'Missing {src}; place v2 checkpoint discovery script alongside the NVDEC profiler.')
    spec=importlib.util.spec_from_file_location(name,src)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def summary(vals):
    if not vals:return {'median_ms':None,'mean_ms':None,'p95_ms':None,'min_ms':None,'max_ms':None}
    vals=sorted(float(x) for x in vals)
    return {'median_ms':round(statistics.median(vals),3),
            'mean_ms':round(statistics.mean(vals),3),
            'p95_ms':round(interp_pct(vals,95),3),
            'min_ms':round(vals[0],3),'max_ms':round(vals[-1],3)}


def interp_pct(v,p):
    x=(len(v)-1)*p/100
    a=int(x);b=min(a+1,len(v)-1)
    return v[a]*(b-x)+v[b]*(x-a)


def hardware_decoder(path):
    """Always request device-memory NVIDIA Video Codec SDK decode; no fallback."""
    import PyNvVideoCodec as nvc
    return nvc.SimpleDecoder(str(path), gpu_id=0, use_device_memory=True,
                             output_color_type=nvc.OutputColorType.RGBP,
                             need_scanned_stream_metadata=True)


def decode_gpu(path, frames):
    """Returns torch [T,3,H,W] uint8 RGB CUDA, plus live decoder for lifetime."""
    import torch
    import numpy as np
    decoder=hardware_decoder(path)
    n=len(decoder)
    if n<1:raise RuntimeError(f'No readable frames in {path}')
    # Match training's np.linspace(...).round().astype(int), including duplicates.
    inds=np.linspace(0,n-1,frames).round().astype(int).tolist()
    unique=list(dict.fromkeys(inds))
    decoded=decoder.get_batch_frames_by_index(unique)
    if len(decoded)!=len(unique):
        raise RuntimeError(f'NVDEC returned {len(decoded)} frames for {len(unique)} distinct indices: {path}')
    tensors=[]
    for f in decoded:
        t=torch.from_dlpack(f)
        if t.device.type!='cuda':
            raise RuntimeError('Decoded frame is not CUDA-resident. Refusing to time a CPU decode as NVDEC GPU path.')
        if t.ndim!=3 or t.shape[0]!=3:
            raise RuntimeError(f'Expected RGBP planar CUDA frame [3,H,W], got {tuple(t.shape)}')
        tensors.append(t)
    indices={k:i for i,k in enumerate(unique)}
    stacked=torch.stack([tensors[indices[i]] for i in inds],dim=0)
    if stacked.dtype!=torch.uint8:
        raise RuntimeError(f'Expected 8-bit source frames, got {stacked.dtype}; inspect input bit depth')
    # decoder/decoded held until after tensors have been safely stacked and used.
    return stacked,decoder,decoded,n


def preprocess_rgbp(rgb,cfg,mean,std):
    import torch
    import torch.nn.functional as F
    # GPU [T,3,H,W] in 0..255. Letterbox matches source aspect/placement.
    if rgb.ndim!=4 or rgb.shape[1]!=3:raise ValueError('Bad decoded RGBP video tensor shape')
    frames,c,ih,iw=rgb.shape
    w,h=int(cfg['width']),int(cfg['height'])
    scale=min(w/iw,h/ih)
    nw=min(w,max(1,round(iw*scale)))
    nh=min(h,max(1,round(ih*scale)))
    x=rgb.float()
    if (iw,ih)!=(nw,nh):
        if scale<1:
            x=F.interpolate(x,size=(nh,nw),mode='area')
        else:
            x=F.interpolate(x,size=(nh,nw),mode='bilinear',align_corners=False)
        # Training script letterboxes uint8, then normalizes. Quantize to match.
        x=x.round().clamp_(0,255)
    l=(w-nw)//2;top=(h-nh)//2
    if (nw,nh)!=(w,h):
        x=F.pad(x,(l,w-nw-l,top,h-nh-top),mode='constant',value=0)
    x=x.unsqueeze(0).div_(255.0)
    x=((x-mean)/std).permute(0,2,1,3,4).contiguous()
    if tuple(x.shape)!=(1,3,frames,h,w):
        raise RuntimeError(f'Incorrect model input {tuple(x.shape)}')
    return x


def process_one(video,cfg,model,amp_dtype,mean,std,detail=False):
    """One complete file->NVDEC->GPU->inference->host decision pipeline."""
    import torch
    torch.cuda.synchronize()
    stamps={'start':time.perf_counter()}
    video_tensor, decoder, live_frames, source_frame_count = decode_gpu(video,int(cfg['frames']))
    if detail:
        torch.cuda.synchronize(); stamps['decoded']=time.perf_counter()
    x=preprocess_rgbp(video_tensor,cfg,mean,std)
    if detail:
        torch.cuda.synchronize(); stamps['prepared']=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=amp_dtype):
        logits=model(x)
    if tuple(logits.shape)!=(1,2):raise RuntimeError(f'Expected [1,2] logits, got {tuple(logits.shape)}')
    if not torch.isfinite(logits).all().item():raise RuntimeError('Non-finite logits')
    pred=int(logits.float().argmax(1).item())
    torch.cuda.synchronize()
    stamps['end']=time.perf_counter()
    # Readable output on CPU included (pred). Model+decode objects are released
    # after the timed path; CUDA is synchronized before dropping decode buffers.
    del x,video_tensor,live_frames,decoder,logits
    result={'e2e_ms':(stamps['end']-stamps['start'])*1000.0,'pred':pred,
            'source_frames':source_frame_count}
    if detail:
        result.update(decode_ms=(stamps['decoded']-stamps['start'])*1000,
                      gpu_preprocess_ms=(stamps['prepared']-stamps['decoded'])*1000,
                      inference_output_ms=(stamps['end']-stamps['prepared'])*1000)
    return result


def selected_manifest_file(args):
    # Profiles with different requested sample sizes cannot silently reuse a set.
    return args.results / f'nvdec_compatible_manifest_{args.videos}.json'


def validate_manifest(obj,args):
    import torch
    if obj.get('manifest_version')!=2:
        raise RuntimeError('Old or invalid NVDEC manifest. Run --stage prepare --rebuild-manifest.')
    if obj.get('device') != torch.cuda.get_device_name(0):
        raise RuntimeError('Manifest was prepared on a different GPU. Use --rebuild-manifest.')
    if obj.get('data') != str(args.data) or obj.get('requested_videos') != args.videos:
        raise RuntimeError('NVDEC manifest belongs to another dataset/sample count.')
    paths=[]
    for record in obj.get('selected',[]):
        path=Path(record['path'])
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise RuntimeError('Manifest input changed/missing: '+str(path))
        paths.append(str(path))
    if len(paths) != args.videos or len(set(paths)) != len(paths):
        raise RuntimeError('Incomplete/duplicate NVDEC manifest; rebuild explicitly.')
    return paths


def prepare_manifest(args, force=False):
    """Fix one hardware-decodable clip set for ALL models, without CPU fallback.

    The actual NVDEC decode for the maximum requested frame count validates
    capability and source metadata before benchmarking. Unsupported streams
    are not transcoded or silently decoded with CPU; they are logged and
    replaced deterministically by the nearest eligible held-out clip.
    """
    import torch
    import numpy as np
    output=selected_manifest_file(args)
    if force:
        completed=list(args.results.glob('*_nvdec_e2e.json'))
        if completed:
            raise RuntimeError('Cannot rebuild NVDEC manifest while prior model results exist. '
                               'Move/remove the existing *_nvdec_e2e.json files first to avoid mixing benchmarks.')
    if output.is_file() and not force:
        obj=json.loads(output.read_text())
        result=validate_manifest(obj,args)
        print('REUSING NVDEC MANIFEST:',output,'videos',len(result),flush=True)
        return result
    parent=parent_module()
    # Use a single established original runner for the source-group test listing.
    source,_,_,_=parent.load_source('x3d_s',args)
    all_pairs=source.scan(args.data,'test')
    if len(all_pairs)!=232:
        raise RuntimeError(f'Expected 232 source-separated SCAD test clips, got {len(all_pairs)}')
    files=[Path(pair[0]).resolve() for pair in all_pairs]
    # Current 17 configs require up to 32 sampled frames. Verify all selected videos at this maximum.
    max_frames=32
    # Primary target locations uniformly cover the original sorted test split.
    # Nearby replacements preserve spread when an unsupported video occurs.
    targets=np.linspace(0,len(files)-1,args.videos).round().astype(int).tolist()
    used=set(); tested={}; rejected=[]; chosen=[]
    for n,target in enumerate(targets,1):
        order=sorted(range(len(files)),key=lambda i:(abs(i-target),i))
        for idx in order:
            if idx in used: continue
            if idx not in tested:
                path=files[idx]
                try:
                    # Actual indexed NVDEC decode, not merely a codec-string check.
                    rgb,decoder,frames,source_n=decode_gpu(path,max_frames)
                    torch.cuda.synchronize()
                    if rgb.device.type!='cuda': raise RuntimeError('NVDEC frames are not on CUDA')
                    tested[idx]=(True,int(source_n),tuple(rgb.shape))
                    del rgb,decoder,frames
                except Exception as e:
                    reason=str(e).replace('\n',' ').strip()[:380]
                    tested[idx]=(False,reason,None)
                    rejected.append({'path':str(path),'index':idx,'error':reason})
                    print('SKIP UNSUPPORTED NVDEC',idx,path.name,reason[:150],flush=True)
            if tested[idx][0]:
                used.add(idx);chosen.append(idx)
                print(f'SELECTED {n}/{args.videos}: {files[idx].name} (dataset index {idx}, frames {tested[idx][1]})',flush=True)
                break
        else:
            raise RuntimeError(f'Only {len(chosen)} compatible files found among {len(files)} test clips. '
                               'NVDEC has no CPU fallback; use fewer --videos or transcode a separate benchmark copy.')
    records=[{'path':str(files[i]),'dataset_index':i,
              'bytes':files[i].stat().st_size,'source_frames':tested[i][1]}
             for i in chosen]
    obj={'manifest_version':2,'data':str(args.data),
         'requested_videos':args.videos,'device':torch.cuda.get_device_name(0),
         'validation_frames':max_frames,'selection_rule':'uniformly spread test indices, nearest successfully NVDEC-tested replacement',
         'selected':records,'rejected':rejected,
         'tested_count':len(tested),'excluded_count':len(rejected),
         'nvdec_only':True,'no_cpu_fallback':True}
    output.write_text(json.dumps(obj,indent=2))
    print('SAVED NVDEC MANIFEST',output,'selected',len(records),'rejected',len(rejected),flush=True)
    return validate_manifest(obj,args)


def select_videos(source,data,videos):
    raise RuntimeError('Legacy video selection disabled: all models must use an NVDEC-verified common manifest.')


def run_check(args):
    import torch
    import PyNvVideoCodec as nvc
    import numpy as np
    if not torch.cuda.is_available():raise RuntimeError('CUDA not detected')
    print('GPU',torch.cuda.get_device_name(0))
    print('PyNvVideoCodec',getattr(nvc,'__version__','installed (version unavailable)'))
    print('NVDEC mode = PyNvVideoCodec SimpleDecoder(use_device_memory=True, RGBP), no CPU fallback')
    parent=parent_module()
    files=sorted((p for p in (args.data/'test').rglob('*') if p.is_file() and p.suffix.lower() in {'.mp4','.mkv','.avi','.mov'}))
    if not files:raise RuntimeError('No supported container found under test/')
    path=files[0]
    t0=time.perf_counter()
    rgb,dec,live,n=decode_gpu(path,8)
    torch.cuda.synchronize()
    print('NVDEC CHECK PASS:',path,'| source frames',n,'| decoded',tuple(rgb.shape),'| GPU',rgb.device,
          '| wall_ms',round(1000*(time.perf_counter()-t0),2))
    print('MODELS AVAILABLE',len(parent.SOURCES),'| checkpoint audit follows:')
    try:
        parent.audit(args)
    except Exception as exc:
        print('AUDIT WARNING:',exc)


def run_single(args):
    import numpy as np
    import torch
    import PyNvVideoCodec as nvc
    import gc
    parent=parent_module()
    name=args.single
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(min(2,os.cpu_count() or 2))
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    path=parent.discover(name,args)
    source,inner,cfg,source_dir=parent.load_source(name,args)
    selected=prepare_manifest(args,force=args.rebuild_manifest)
    print('MODEL',name,'| checkpoint',path,'| input',cfg['frames'],cfg['width'],cfg['height'],flush=True)
    print('NVDEC GPU-resident RGBP + GPU letterbox + normalization + inference | no CPU image transfer',flush=True)
    model,initial=source.build_model(inner,source_dir)
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    parent.validate_checkpoint_identity(name,ckpt,cfg)
    model.load_state_dict(parent.state_of(ckpt),strict=True)
    model=model.cuda().eval()
    params=sum(p.numel() for p in model.parameters())
    mean=torch.tensor(cfg['mean'],device='cuda',dtype=torch.float32).view(1,1,3,1,1)
    std=torch.tensor(cfg['std'],device='cuda',dtype=torch.float32).view(1,1,3,1,1)
    torch.cuda.reset_peak_memory_stats()
    # First run: verify actual NVDEC+model path and allow kernels to initialize.
    for i in range(args.warmup):
        sample=selected[i%len(selected)]
        trial=process_one(sample,cfg,model,amp_dtype,mean,std)
        print('WARMUP',i+1,'/',args.warmup,round(trial['e2e_ms'],2),'ms',flush=True)
    times=[];decisions=[];samples=[]
    for i in range(args.repeats):
        video=selected[i%len(selected)]
        res=process_one(video,cfg,model,amp_dtype,mean,std)
        times.append(res['e2e_ms']);decisions.append(res['pred']);samples.append(video)
        if (i+1)%max(1,args.repeats//5)==0:
            print('TIMED',i+1,'/',args.repeats,'last_ms',round(times[-1],2),flush=True)
    # Separate diagnostics, intentionally NOT substituted into main timings.
    stage=[]
    for i in range(args.breakdown_samples):
        video=selected[i%len(selected)]
        stage.append(process_one(video,cfg,model,amp_dtype,mean,std,detail=True))
    rec=dict(model=name,checkpoint=str(path),parameters_m=round(params/1e6,4),
             frames=int(cfg['frames']),input_width=int(cfg['width']),input_height=int(cfg['height']),
             gpu=torch.cuda.get_device_name(0),precision=str(amp_dtype).replace('torch.',''),
             backend='NVIDIA PyNvVideoCodec SimpleDecoder',decode_hardware='NVDEC',
             output_color='RGBP device-memory (CUDA DLPack)',
             gpu_preprocess='letterbox area/bilinear, black pad, training mean/std',
             benchmark='whole-clip batch 1 serial; decoder constructed for each clip',
             includes=['file access','container demux','NVDEC indexed sampling','RGB conversion',
                       'GPU tensor stacking','GPU resize and letterbox','GPU normalization',
                       'classification forward','prediction GPU-to-host','CUDA synchronization'],
             warmup=args.warmup,repeats=args.repeats,unique_video_count=len(set(selected)),
             video_manifest=str(selected_manifest_file(args)),
             e2e=summary(times),
             separate_breakdown_diagnostic={
                 'decode_and_open':summary([s['decode_ms'] for s in stage]),
                 'gpu_preprocess':summary([s['gpu_preprocess_ms'] for s in stage]),
                 'inference_and_output':summary([s['inference_output_ms'] for s in stage])} if stage else None,
             inference_gpu_peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2,2),
             warning='CUDA preprocessing uses torch interpolation; not guaranteed bit-identical to training OpenCV. This is a latency result, not a re-evaluation of test accuracy.')
    out=args.results/(name+'_nvdec_e2e.json')
    out.write_text(json.dumps(rec,indent=2))
    print('NVDEC RESULT',name,json.dumps(rec['e2e']),flush=True)
    print('OUTPUT',out,flush=True)
    del model,ckpt,mean,std,source
    gc.collect();torch.cuda.empty_cache()
    return rec


def save_csv(args):
    parent=parent_module()
    rows=[]
    for name in parent.SOURCES:
        f=args.results/(name+'_nvdec_e2e.json')
        if not f.exists():continue
        d=json.loads(f.read_text())
        row={'model':name,'parameters_m':d['parameters_m'],'frames':d['frames'],
             'width':d['input_width'],'height':d['input_height'],
             'nvdec_e2e_median_ms':d['e2e']['median_ms'],
             'nvdec_e2e_mean_ms':d['e2e']['mean_ms'],
             'nvdec_e2e_p95_ms':d['e2e']['p95_ms'],
             'nvdec_e2e_min_ms':d['e2e']['min_ms'],
             'nvdec_e2e_max_ms':d['e2e']['max_ms'],
             'gpu_peak_allocated_mib':d['inference_gpu_peak_allocated_mib'],
             'repeats':d['repeats'],'precision':d['precision']}
        if d.get('separate_breakdown_diagnostic'):
            for name2,label in [('decode_and_open','nvdec_decode'),('gpu_preprocess','gpu_preprocess'),('inference_and_output','model_output')]:
                row[label+'_diagnostic_median_ms']=d['separate_breakdown_diagnostic'][name2]['median_ms']
        rows.append(row)
    if rows:
        keys=list(rows[0].keys())
        with (args.results/'scad17_nvdec_e2e_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
            writer.writeheader();writer.writerows(rows)
        print('SUMMARY',args.results/'scad17_nvdec_e2e_summary.csv','(',len(rows),'models)',flush=True)


def run_all(args):
    # Preselect once, before launching individual isolated model processes.
    prepare_manifest(args,force=args.rebuild_manifest)
    parent=parent_module()
    fail=[]
    for name in parent.SOURCES:
        dest=args.results/(name+'_nvdec_e2e.json')
        if dest.exists() and not args.rerun:
            print('SKIPPING completed:',name,flush=True)
            continue
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage','single','--single',name,
             '--home',str(args.home),'--data',str(args.data),'--checkpoints',str(args.checkpoints),
             '--results',str(args.results),'--warmup',str(args.warmup),'--repeats',str(args.repeats),
             '--videos',str(args.videos),'--breakdown-samples',str(args.breakdown_samples)]
        print('\nNVDEC E2E PROFILE',name,flush=True)
        with (args.results/(name+'_nvdec.log')).open('w') as f:
            proc=subprocess.Popen(cmd,cwd=args.home,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line);f.write(line)
            ret=proc.wait()
        if ret:
            fail.append(name)
            print('FAILED',name,'see',args.results/(name+'_nvdec.log'),flush=True)
        save_csv(args)
    if fail:
        (args.results/'nvdec_failures.json').write_text(json.dumps(fail,indent=2))
        raise RuntimeError('Some NVDEC profiles failed: '+', '.join(fail))
    print('ALL DONE',args.results/'scad17_nvdec_e2e_summary.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description='NVIDIA NVDEC end-to-end GPU latency for 17 SCAD configurations')
    p.add_argument('--stage',choices=['check','prepare','profile','single'],default='check')
    p.add_argument('--single',default=None)
    p.add_argument('--home',type=Path,default=DEFAULT_HOME)
    p.add_argument('--data',type=Path,default=None)
    p.add_argument('--checkpoints',type=Path,default=None)
    p.add_argument('--results',type=Path,default=None)
    p.add_argument('--videos',type=int,default=16,help='Fixed set spread across the held-out split')
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--repeats',type=int,default=40)
    p.add_argument('--breakdown-samples',type=int,default=8)
    p.add_argument('--rerun',action='store_true')
    p.add_argument('--rebuild-manifest',action='store_true',help='Explicitly replace fixed NVDEC-compatible video sample manifest')
    args=p.parse_args()
    args.home=args.home.expanduser().resolve()
    args.data=(args.data or args.home/'accident').expanduser().resolve()
    args.checkpoints=(args.checkpoints or args.home/'scad'/'models').expanduser().resolve()
    args.results=(args.results or args.home/'scad'/'profiling_17'/'nvdec_e2e').expanduser().resolve()
    args.results.mkdir(parents=True,exist_ok=True)
    if args.repeats<1 or args.warmup<0 or args.videos<1 or args.videos>232 or args.breakdown_samples<0:
        p.error('warmup/repeats/videos/breakdown-samples out of range')
    parent=parent_module()
    if args.single and args.single not in parent.SOURCES:p.error('Unknown --single: '+args.single)
    if args.stage=='single' and not args.single:p.error('--single MODEL is required')
    if args.stage=='check':run_check(args)
    elif args.stage=='prepare':prepare_manifest(args,force=args.rebuild_manifest)
    elif args.stage=='single':run_single(args)
    elif args.stage=='profile':run_all(args)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('NVDEC PIPELINE FAILED:',exc,file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.exit(2)
