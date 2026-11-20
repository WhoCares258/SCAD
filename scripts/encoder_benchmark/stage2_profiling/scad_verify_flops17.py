#!/usr/bin/env python3
"""Audit forward-pass operation counts for the 17 saved SCAD configurations.

This is not training or latency measurement. Uses original model builders and
strictly loaded fine-tuned checkpoints; one synthetic B=1 input/config.
Counts multiply + add as two FLOPs, and reports the 1-MAC equivalent separately.
Unsupported/unregistered operations are not silently called 'verified'.
"""
from __future__ import annotations
import argparse, csv, gc, inspect, json, os, subprocess, sys, traceback
from pathlib import Path
from types import SimpleNamespace

HOME = Path('/home/user/22104053')
# Exact historical screening checkpoints, not the 50 new seed checkpoints.
# Avoid ambiguous discovery after five-seed training created duplicate basenames.
CHECKPOINT_HINTS = {
 'videomae_v1_l':'scad/videomae_v1_l/scad_variant_videomae_v1_l_best.pt',
 'vjepa21_l':'scad/vjepa21_l/scad_variant_vjepa21_l_best.pt',
 'vjepa21_b_full':'scad/vjepa21_full/scad_next_vjepa21_full_best.pt',
 'internvideo2_l':'scad/internvideo2_l/scad_variant_internvideo2_l_best.pt',
 'mvitv2_b':'scad/mvitv2_b/scad_variant_mvitv2_b_best.pt',
 'videomaev2_b':'scad/videomaev2_b/scad6_videomaev2_b_best.pt',
 'videomaev2_s':'scad/videomaev2_s/scad_variant_videomaev2_s_best.pt',
 'mvitv2_s':'scad/mvitv2_s/scad6_mvitv2_s_best.pt',
 'videomaev2_l':'scad/videomaev2_l/scad_variant_videomaev2_l_best.pt',
 'videomae_v1_b':'scad/videomae_v1_b/scad_variant_videomae_v1_b_best.pt',
 'mvitv2_l_maskfeat16':'scad_variant_mvitv2_l_maskfeat16_best.pt',
 'internvideo2_b':'scad/internvideo2_b/scad_next_internvideo2_b_best.pt',
 'internvideo2_s':'scad/internvideo2_s/scad_variant_internvideo2_s_best.pt',
 'x3d_m':'scad/models/x3d_m/scad_variant_x3d_m_best.pt',
 'x3d_s_480x360':'scad/models/x3d_s_480x360/scad6_x3d_s_best.pt',
 'x3d_s':'scad/models/x3d_s/scad_next_x3d_s_original_best.pt',
 'slowfast_r50':'scad/models/slowfast_r50/scad_variant_slowfast_r50_best.pt',
}

def pipeline():
    from importlib.util import spec_from_file_location, module_from_spec
    f=Path(__file__).resolve().parent/'scad_17_pipeline_v2.py'
    if not f.exists():f=HOME/'scad_17_pipeline_v2.py'
    if not f.exists():raise FileNotFoundError('Copy scad_17_pipeline_v2.py to the same folder as this script.')
    sp=spec_from_file_location('scad_profiler_17_original',f)
    m=module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
    return m

def paths(args):
    return SimpleNamespace(home=args.home,checkpoints=args.home/'scad'/'models',
                           data=args.home/'accident',results=args.out)

def one(args):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    m=pipeline(); cfg_paths=paths(args)
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required for the exact configured model forward pass')
    name=args.single
    ckpt=args.home/CHECKPOINT_HINTS[name]
    if not ckpt.is_file():
        raise FileNotFoundError(f'Missing exact original screening checkpoint: {ckpt}; do not substitute a new five-seed checkpoint')
    mod,inner,cfg,sources=m.load_source(name,cfg_paths)
    torch.manual_seed(42)
    model,_=mod.build_model(inner,sources)
    saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    m.validate_checkpoint_identity(name,saved,cfg)
    model.load_state_dict(m.state_of(saved),strict=True)
    model=model.cuda().eval()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    T=int(cfg['frames']);H=int(cfg['height']);W=int(cfg['width'])
    # Original model-only profiler feeds normalized B,C,T,H,W at the specified size.
    inp=torch.zeros((1,3,T,H,W),device='cuda',dtype=torch.float32)
    with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
        z=model(inp)
    if tuple(z.shape)!=(1,2):raise RuntimeError(f'Unexpected logits shape {tuple(z.shape)}')
    torch.cuda.synchronize()
    opts={'display':False}
    if 'skip_unsupported' in inspect.signature(FlopCounterMode).parameters:
        opts['skip_unsupported']=True
    with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
        with FlopCounterMode(**opts) as counter:
            result=model(inp)
            if tuple(result.shape)!=(1,2):raise RuntimeError('Incorrect logits during FLOPs audit')
        torch.cuda.synchronize()
    gflops=int(counter.get_total_flops())/1e9
    counts=counter.get_flop_counts().get('Global',{})
    ops={str(k):round(float(v)/1e9,5) for k,v in counts.items()}
    getter=getattr(counter,'get_unsupported_ops',None)
    unsupported={str(k):int(v) for k,v in getter().items()} if callable(getter) else None
    convolution=sum(v for k,v in ops.items() if 'convolution' in k)
    attention=sum(v for k,v in ops.items() if 'attention' in k or 'flash' in k)
    rec={'model':name,'checkpoint':str(ckpt),'input_frames':T,'height':H,'width':W,
         'parameters_m':round(sum(p.numel() for p in model.parameters())/1e6,5),
         'forward_gflops_2_per_mac':round(gflops,5),
         'gmac_equivalent_1_per_mac':round(gflops/2,5),
         'counted_conv_gflops':round(convolution,5),
         'counted_attention_gflops':round(attention,5),
         'counted_ops_gflops':ops,'unsupported_ops':unsupported,
         'coverage_note':'Operator formulas count matrix ops, conv2d/3d and registered SDPA. Other operations/custom kernels may be omitted. Theoretical forward FLOPs, not GPU instruction count or latency.',
         'precision':str(dtype),'device':torch.cuda.get_device_name(0),
         'torch_version':torch.__version__}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/(name+'_flops.json')).write_text(json.dumps(rec,indent=2,default=str))
    print('GFLOPS_RESULT',json.dumps({k:v for k,v in rec.items() if k not in ('counted_ops_gflops','coverage_note','checkpoint')}),flush=True)
    del model,saved,inp,z,result
    gc.collect();torch.cuda.empty_cache()
    return rec

def summary(args):
    m=pipeline()
    records=[]
    for n in m.SOURCES:
        p=args.out/(n+'_flops.json')
        if p.is_file():records.append(json.loads(p.read_text()))
    args.out.mkdir(parents=True,exist_ok=True)
    cols=['model','input_frames','height','width','parameters_m','forward_gflops_2_per_mac',
          'gmac_equivalent_1_per_mac','counted_conv_gflops','counted_attention_gflops',
          'unsupported_ops','precision','torch_version']
    with (args.out/'scad17_flops_audit.csv').open('w',newline='') as fp:
        wr=csv.DictWriter(fp,fieldnames=cols,extrasaction='ignore')
        wr.writeheader();wr.writerows(records)
    print(f'GFLOPS_SUMMARY {len(records)}/17: {args.out/"scad17_flops_audit.csv"}',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--all',action='store_true')
    p.add_argument('--single',default=None)
    p.add_argument('--home',type=Path,default=HOME)
    p.add_argument('--out',type=Path,default=HOME/'scad'/'flops_verified_17')
    p.add_argument('--rerun',action='store_true')
    args=p.parse_args();args.home=args.home.expanduser().resolve();args.out=args.out.expanduser().resolve()
    m=pipeline()
    if args.single and args.single not in m.SOURCES: p.error('Unknown model '+args.single)
    if args.all:
        errors={}
        for n in m.SOURCES:
            if (args.out/(n+'_flops.json')).exists() and not args.rerun:
                print('SKIP EXISTING',n,flush=True);continue
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--single',n,
                 '--home',str(args.home),'--out',str(args.out)]
            print('AUDIT',n,flush=True)
            ret=subprocess.run(cmd,cwd=args.home)
            if ret.returncode:
                errors[n]=ret.returncode
                print('ERROR',n,ret.returncode,flush=True)
        summary(args)
        if errors:raise SystemExit('INCOMPLETE: '+str(errors))
    elif args.single:one(args);summary(args)
    else:summary(args)

if __name__=='__main__': main()
