#!/usr/bin/env python3
"""Final SCAD checkpoint benchmark: 10 NVDEC-trained models x seeds 42..46.

NO TRAINING. Uses the exact saved SCAD 50-run checkpoints, the final10 dataset
manifest (including its 18 offline transcoded streams), the original model
builders, and the prior uninstrumented cold-clip NVDEC E2E implementation.
Each model/seed runs in its own process. Three passes x 232 test clips, B=1.
Accuracy from the same final checkpoint and the same E2E pipeline; original
training-time results are preserved untouched for comparison.

Place next to scad_nvdec_final10.py and scad_17_nvdec_e2e_FIXED.py.
"""
from __future__ import annotations
import argparse, csv, gc, json, math, os, statistics, subprocess, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace

import scad_nvdec_final10 as train
import scad_17_nvdec_e2e_FIXED as nv

DEFAULT_HOME=Path('/home/user/22104053')
SEEDS=(42,43,44,45,46)
FIELDS=['model','seed','frames','width','height','parameters_m','accuracy_pct','macro_f1_pct',
        'correct','training_accuracy_pct','batch1_minus_training_pp','e2e_mean_ms',
        'e2e_median_ms','e2e_p95_ms','pass1_mean_ms','pass2_mean_ms','pass3_mean_ms',
        'n_clips','n_timing_samples','checkpoint','device','precision']

def percentile(vals,p):
    vals=sorted(vals);k=(len(vals)-1)*p/100;a=int(k);b=min(a+1,len(vals)-1)
    return vals[a]+(vals[b]-vals[a])*(k-a)

def metrics(ys,ps):
    from sklearn.metrics import f1_score, confusion_matrix
    n=len(ys);correct=sum(a==b for a,b in zip(ys,ps))
    return {'accuracy_pct':100*correct/n,'macro_f1_pct':100*f1_score(ys,ps,average='macro',zero_division=0),
            'correct':correct,'confusion_matrix':confusion_matrix(ys,ps,labels=[0,1]).tolist()}

def atomic_json(path,object_):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix('.json.tmp')
    t.write_text(json.dumps(object_,indent=2,default=str));os.replace(t,path)

def load_manifest(args):
    shim=SimpleNamespace(out=args.training_out,data=args.data,crf=args.crf)
    obj=train.manifest_read(shim,full_validate=False)
    clips=sorted((r for r in obj['records'] if r['split']=='test'),key=lambda r:r['index'])
    if len(clips)!=232 or [r['index'] for r in clips]!=list(range(232)):
        raise RuntimeError('Expected indexed 232-video final10 test manifest')
    if sorted([r['label'] for r in clips]) != [0]*116+[1]*116:
        raise RuntimeError('Test label distribution changed')
    for r in clips:
        if not Path(r['prepared']).is_file() or Path(r['prepared']).stat().st_size!=r['prepared_size']:
            raise RuntimeError('Prepared video missing/modified: '+r['prepared'])
        if not Path(r['original']).is_file() or Path(r['original']).stat().st_size!=r['original_size']:
            raise RuntimeError('Original video missing/modified: '+r['original'])
    return clips

def locate(args,name,seed):
    if name not in train.SPECS or seed not in SEEDS:raise ValueError('Unknown model or seed')
    stamp=args.training_out/'runs'/name/f'seed_{seed}'/'final10_complete.json'
    if not stamp.is_file():raise FileNotFoundError('Missing certified completed run: '+str(stamp))
    record=json.loads(stamp.read_text())
    if record.get('model')!=name or int(record.get('seed',-1))!=seed or record.get('pipeline')!=train.VERSION:
        raise ValueError('Completion marker identity mismatch: '+str(stamp))
    checkpoint=Path(record['checkpoint'])
    if not checkpoint.is_file():
        # Only accept the exact original checkpoint filename in its certified seed directory.
        alternative=stamp.parent/checkpoint.name
        if not alternative.is_file():raise FileNotFoundError('Certified checkpoint unavailable: '+str(checkpoint))
        checkpoint=alternative
    if checkpoint.resolve().parent!=stamp.parent.resolve():
        raise RuntimeError('Checkpoint outside the certified run folder; refusing ambiguous substitution: '+str(checkpoint))
    return stamp,checkpoint,record

def audit(args):
    clips=load_manifest(args)
    print('SCAD FINAL NVDEC E2E checkpoint audit: test clips',len(clips),'models',len(train.SPECS),'seeds',len(SEEDS))
    for name in train.SPECS:
        for seed in SEEDS:
            _,ckpt,record=locate(args,name,seed)
            print('CHECKPOINT OK',name,'seed',seed,'bytes',ckpt.stat().st_size,
                  'training_acc',record['test_accuracy_pct'])
    print('AUDIT PASS 50/50; no model training needed.')

def run_one(args):
    import torch
    name=args.model;seed=args.seed
    _,ckptpath,record=locate(args,name,seed)
    clips=load_manifest(args)
    result_path=args.results/'runs'/name/f'seed_{seed}'/'complete.json'
    if result_path.exists() and not args.rerun:
        print('SKIP VERIFIED EXISTING:',result_path,flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA GPU required for NVDEC')
    torch.set_num_threads(min(2,os.cpu_count() or 2))
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(42)  # model builder initialization, checkpoint overwrites weights strictly
    source,inner=train.source_module(name)
    cfg=dict(source.SPECS[inner]);nframes=int(cfg['frames']);width=int(cfg['width']);height=int(cfg['height'])
    if (nframes,width,height)!=(record['frames'],record['width'],record['height']):
        raise RuntimeError('Final checkpoint input specification differs from training record')
    home_args=SimpleNamespace(home=args.home)
    model,_=source.build_model(inner,train.source_dir(name,home_args))
    saved=torch.load(ckptpath,map_location='cpu',weights_only=False)
    meta=saved.get('configuration',{}) if isinstance(saved,dict) else {}
    if isinstance(meta,dict):
        if meta.get('model') not in (None,inner):raise RuntimeError('Checkpoint model ID does not match original builder')
        for key,target in (('frames',nframes),('input_width',width),('input_height',height)):
            if key in meta and int(meta[key])!=target:raise RuntimeError('Checkpoint metadata mismatch: '+key)
    state=saved.get('model') if isinstance(saved,dict) else None
    if not isinstance(state,dict):raise RuntimeError('Fine-tuned checkpoint missing model state dict')
    model.load_state_dict(state,strict=True)
    model=model.cuda().eval()
    params=sum(p.numel() for p in model.parameters())/1e6
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    mean=torch.tensor(cfg['mean'],device='cuda',dtype=torch.float32).view(1,1,3,1,1)
    std=torch.tensor(cfg['std'],device='cuda',dtype=torch.float32).view(1,1,3,1,1)
    print('BENCH',name,'SEED',seed,'CKPT',ckptpath,'INPUT',nframes,width,height,
          'PARAMS_M',round(params,4),'DType',str(dtype),flush=True)
    # Warmups use real prepared clips but are not included in timing distributions.
    for j in range(args.warmup):
        nv.process_one(clips[j % len(clips)]['prepared'],cfg,model,dtype,mean,std)
    ys=[int(r['label']) for r in clips]
    passes=[];samples=[]
    for p in range(1,args.passes+1):
        predictions=[None]*len(clips)
        times=[]
        order=range(len(clips)) if p%2 else range(len(clips)-1,-1,-1)
        for k,i in enumerate(order,1):
            rec=clips[i]
            out=nv.process_one(rec['prepared'],cfg,model,dtype,mean,std,detail=False)
            predictions[i]=int(out['pred'])
            ms=float(out['e2e_ms'])
            if not math.isfinite(ms) or ms<=0:raise RuntimeError('Non-finite/nonpositive E2E duration')
            times.append(ms)
            samples.append({'pass':p,'index':i,'label':ys[i], 'prediction':predictions[i],
                            'e2e_ms':round(ms,5),'prepared_type':rec['processing']})
            if k%58==0:print('PASS',p,'/',args.passes,k,'/232',flush=True)
        passes.append({'number':p,'predictions':predictions,'mean_ms':statistics.mean(times),
                       'median_ms':statistics.median(times),'p95_ms':percentile(times,95)})
        print('PASS COMPLETE',p,'MEAN',round(passes[-1]['mean_ms'],3),'ACC',round(metrics(ys,predictions)['accuracy_pct'],4),flush=True)
    if any(z['predictions']!=passes[0]['predictions'] for z in passes[1:]):
        raise RuntimeError('Prediction differences across identical full-set timing passes; inspect determinism')
    accurate=metrics(ys,passes[0]['predictions'])
    all_ms=[s['e2e_ms'] for s in samples]
    new=dict(model=name,seed=seed,checkpoint=str(ckptpath),frames=nframes,width=width,height=height,
             parameters_m=params,device=torch.cuda.get_device_name(0),precision=str(dtype),
             n_clips=len(clips),n_timing_samples=len(samples),
             e2e_mean_ms=statistics.mean(all_ms),e2e_median_ms=statistics.median(all_ms),
             e2e_p95_ms=percentile(all_ms,95),pass1_mean_ms=passes[0]['mean_ms'],
             pass2_mean_ms=passes[1]['mean_ms'],pass3_mean_ms=passes[2]['mean_ms'],
             **accurate, training_accuracy_pct=float(record['test_accuracy_pct']),
             batch1_minus_training_pp=accurate['accuracy_pct']-float(record['test_accuracy_pct']),
             benchmark_definition='232 unique prepared test clips x 3 cold-clip full passes; NVDEC indexed GPU RGBP; GPU letterbox/normalization; BF16/FP16 model inference and host prediction; one clip at a time',
             source_manifest=str(args.training_out/'prepared_manifest.json'),
             offline_transcoding_included_in_e2e=False,
             prepared_transcoded_count=sum(r['processing']=='transcoded_h264' for r in clips),
             per_pass=[{k:v for k,v in z.items() if k!='predictions'} for z in passes])
    result_path.parent.mkdir(parents=True,exist_ok=True)
    # Store per-video observations; completion marker only after all CSVs written.
    with (result_path.parent/'timings.csv').open('w',newline='') as fp:
        wr=csv.DictWriter(fp,fieldnames=list(samples[0]));wr.writeheader();wr.writerows(samples)
    with (result_path.parent/'predictions.csv').open('w',newline='') as fp:
        wr=csv.DictWriter(fp,fieldnames=['index','label','prediction','correct'])
        wr.writeheader()
        for i,(y,pr) in enumerate(zip(ys,passes[0]['predictions'])):
            wr.writerow({'index':i,'label':y,'prediction':pr,'correct':int(y==pr)})
    atomic_json(result_path,new)
    print('CHECKPOINT VERIFIED',name,seed,'ACC',round(new['accuracy_pct'],4),
          'MACRO_F1',round(new['macro_f1_pct'],4),'E2E_MEAN',round(new['e2e_mean_ms'],3),
          'E2E_P95',round(new['e2e_p95_ms'],3),flush=True)
    del model,saved,state,mean,std
    gc.collect();torch.cuda.empty_cache()

def summarize(args):
    rows=[]
    for name in train.SPECS:
        for seed in SEEDS:
            f=args.results/'runs'/name/f'seed_{seed}'/'complete.json'
            if not f.exists():continue
            r=json.loads(f.read_text())
            if (r.get('model'),r.get('seed'))!=(name,seed) or r.get('n_timing_samples')!=232*args.passes:
                raise RuntimeError('Invalid or wrong-protocol completion marker '+str(f))
            rows.append(r)
    args.results.mkdir(parents=True,exist_ok=True)
    with (args.results/'final50_per_seed_accuracy_latency.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=FIELDS,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    bymodel=[]
    for name in train.SPECS:
        subset=[r for r in rows if r['model']==name]
        if not subset:continue
        obj={'model':name,'n_seeds':len(subset)}
        for v in ('accuracy_pct','macro_f1_pct','e2e_mean_ms','e2e_median_ms','e2e_p95_ms'):
            obj[f'mean_{v}']=statistics.mean(r[v] for r in subset)
            obj[f'sd_{v}']=statistics.stdev(r[v] for r in subset) if len(subset)>1 else None
        obj['mean_original_training_accuracy_pct']=statistics.mean(r['training_accuracy_pct'] for r in subset)
        obj['mean_batch1_minus_training_pp']=statistics.mean(r['batch1_minus_training_pp'] for r in subset)
        bymodel.append(obj)
    bymodel.sort(key=lambda v:v['mean_accuracy_pct'],reverse=True)
    if bymodel:
        with (args.results/'final10_accuracy_latency_mean_sd.csv').open('w',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=list(bymodel[0]));writer.writeheader();writer.writerows(bymodel)
    # True nondominance on mean accuracy and mean E2E (higher accuracy, lower latency).
    frontier=[]
    if all(r['n_seeds']==5 for r in bymodel) and len(bymodel)==10:
        for r in bymodel:
            dominated=any(other is not r and
                          other['mean_accuracy_pct']>=r['mean_accuracy_pct'] and
                          other['mean_e2e_mean_ms']<=r['mean_e2e_mean_ms'] and
                          (other['mean_accuracy_pct']>r['mean_accuracy_pct'] or
                           other['mean_e2e_mean_ms']<r['mean_e2e_mean_ms'])
                          for other in bymodel)
            if not dominated:frontier.append(r)
        frontier.sort(key=lambda r:r['mean_e2e_mean_ms'])
        with (args.results/'final10_pareto_frontier.csv').open('w',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=list(frontier[0]));writer.writeheader();writer.writerows(frontier)
    print('\nCHECKPOINT LATENCY FINISHED',len(rows),'/50',flush=True)
    print(f"{'Model':<23} {'Seeds':>5} {'Acc (%)':>12} {'SD':>8} {'E2E Mean ms':>13} {'SD ms':>9} {'P95 ms':>10}")
    for r in bymodel:
        print(f"{r['model']:<23} {r['n_seeds']:>5} {r['mean_accuracy_pct']:>12.3f} "
              f"{r['sd_accuracy_pct'] if r['sd_accuracy_pct'] is not None else float('nan'):>8.3f} "
              f"{r['mean_e2e_mean_ms']:>13.3f} "
              f"{r['sd_e2e_mean_ms'] if r['sd_e2e_mean_ms'] is not None else float('nan'):>9.3f} "
              f"{r['mean_e2e_p95_ms']:>10.3f}")
    print('CSV',args.results/'final10_accuracy_latency_mean_sd.csv',flush=True)
    if frontier:print('FINAL PARETO:',', '.join(r['model'] for r in frontier),flush=True)

def launch(args):
    audit(args)
    # Original model order; each seed runs in its own fresh process to avoid GPU model memory growth.
    for name in train.SPECS:
        for seed in SEEDS:
            f=args.results/'runs'/name/f'seed_{seed}'/'complete.json'
            if f.exists() and not args.rerun:
                print('SKIP',name,seed,flush=True);continue
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage','one',
                 '--model',name,'--seed',str(seed),'--home',str(args.home),
                 '--data',str(args.data),'--training-out',str(args.training_out),
                 '--results',str(args.results),'--crf',str(args.crf),
                 '--warmup',str(args.warmup),'--passes',str(args.passes)]
            if args.rerun:cmd.append('--rerun')
            log=args.results/'logs'/f'{name}_seed{seed}.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            print('\nPROFILE',name,'SEED',seed,'LOG',log,flush=True)
            with log.open('a',buffering=1) as fp:
                process=subprocess.Popen(cmd,cwd=args.home,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout:print(line,end='',flush=True);fp.write(line)
                ret=process.wait()
            if ret:raise RuntimeError('Profiler failed '+name+' seed '+str(seed)+'; see '+str(log))
            summarize(args)
    summarize(args)

def main():
    p=argparse.ArgumentParser(description='SCAD final50 NVDEC cold-clip checkpoint latency; no training')
    p.add_argument('--stage',required=True,choices=['audit','one','all','summary'])
    p.add_argument('--model',choices=list(train.SPECS))
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--home',type=Path,default=DEFAULT_HOME)
    p.add_argument('--data',type=Path)
    p.add_argument('--training-out',type=Path)
    p.add_argument('--results',type=Path)
    p.add_argument('--passes',type=int,default=3)
    p.add_argument('--warmup',type=int,default=15)
    p.add_argument('--crf',type=int,default=14)
    p.add_argument('--rerun',action='store_true')
    a=p.parse_args()
    a.home=a.home.expanduser().resolve()
    a.data=(a.data or a.home/'accident').expanduser().resolve()
    a.training_out=(a.training_out or a.home/'scad'/'nvdec_final10').expanduser().resolve()
    a.results=(a.results or a.home/'scad'/'final50_checkpoint_latency').expanduser().resolve()
    if a.passes!=3:p.error('Fixed protocol requires 3 timing passes')
    if a.warmup<0:p.error('warmup must be nonnegative')
    if a.stage=='one' and (a.model is None or a.seed is None):p.error('Specify --model and --seed')
    if a.stage=='audit':audit(a)
    elif a.stage=='one':run_one(a)
    elif a.stage=='all':launch(a)
    else:summarize(a)

if __name__=='__main__':
    try:main()
    except Exception as ex:
        print('FINAL50 CHECKPOINT LATENCY ERROR',str(ex),file=sys.stderr,flush=True)
        traceback.print_exc()
        raise SystemExit(2)
