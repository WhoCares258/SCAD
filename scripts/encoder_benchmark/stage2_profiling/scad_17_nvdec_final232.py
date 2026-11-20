#!/usr/bin/env python3
"""SCAD: definitive L40S NVDEC inference benchmark, 17 models x 232 clips x 3 passes.

Expected files in same directory:
  scad_17_pipeline_v2.py, scad_17_nvdec_e2e_FIXED.py, scad_17_sources/

Phases:
  prepare  : scan EXACT SCAD test 232 (116+116), validate NVDEC on all,
             transcode unsupported clips into separate high-quality H.264 files,
             preserve original resolution and check source frame counts;
             freeze an auditable manifest (never edit original SCAD dataset).
  single   : process a chosen model; 3 full cold-clip passes of NVDEC E2E timing;
             one additional untimed-reference and measured stage-diagnostics
             pass for original OpenCV-path prediction comparison.
  profile  : one isolated process per checkpoint, resumable, CSV aggregation.
  audit    : inspect expected checkpoint identities + manifest, no GPU inference.

What E2E includes: file open, decoder initialization, demux/indexed HW decode,
GPU RGB planes, GPU letterbox/normalization, inference, prediction host decision
and CUDA synchronization. Offline compatibility transcoding is excluded and
reported separately: this is NVDEC inference on deployment-ready H.264 assets,
not raw-original-file online transcode latency.

Main latency 3 passes are uninstrumented (only entry/exit synchronized); per-stage
breakdown occurs on a fourth, diagnostic pass with boundary synchronizations.
All timing is serial batch=1, BF16 on supported NVIDIA GPU, no CPU decode fallback
in the timed NVDEC pipeline. Reference OpenCV accuracy is separately assessed
against the original source files (not included in latency).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import gc

HOME=Path('/home/user/22104053')
MANIFEST_VERSION=1
# Prior held-out results (original OpenCV test protocol), including 4 retrained CNNs.
# These are EXPECTATIONS, not values written into the NVDEC metrics.
HISTORICAL_TEST_ACCURACY_PCT={
    'videomae_v1_l':93.10, 'vjepa21_l':92.67, 'vjepa21_b_full':90.95,
    'internvideo2_l':90.52, 'mvitv2_b':89.66, 'videomaev2_b':89.22,
    'videomaev2_s':88.79, 'mvitv2_s':88.36, 'videomaev2_l':88.36,
    'videomae_v1_b':88.36, 'mvitv2_l_maskfeat16':87.93,
    'internvideo2_b':86.64, 'internvideo2_s':85.78,
    'x3d_m':78.02, 'x3d_s_480x360':81.47, 'x3d_s':78.02,
    'slowfast_r50':81.90}

def load_relative(filename, module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]
    src=Path(__file__).resolve().parent/filename
    if not src.is_file():
        raise FileNotFoundError(f'Missing {src}; upload all scripts beside each other')
    spec=importlib.util.spec_from_file_location(module_name,src)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[module_name]=mod
    spec.loader.exec_module(mod)
    return mod

def older():return load_relative('scad_17_nvdec_e2e_FIXED.py','_scad_nvdec_base_final')
def parent():return load_relative('scad_17_pipeline_v2.py','_scad_parent_final')

def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=str))
    os.replace(tmp,path)

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda:fh.read(2**20),b''):
            h.update(block)
    return h.hexdigest()

def run_process(cmd, desc):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode:
        raise RuntimeError(f'{desc} failed (exit {p.returncode}): {p.stderr[-1500:]}')
    return p.stdout

def ffprobe_video(path):
    out=run_process(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,nb_frames,avg_frame_rate',
        '-of','json',str(path)],'ffprobe')
    streams=json.loads(out).get('streams',[])
    if len(streams)!=1:raise RuntimeError(f'Expected one video stream: {path}')
    d=streams[0]
    return {'codec':d.get('codec_name'),'width':int(d.get('width',0)),
            'height':int(d.get('height',0)),
            'nb_frames_reported':d.get('nb_frames'),'fps':d.get('avg_frame_rate')}

def opencv_framecount(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError('OpenCV cannot read original source '+str(path))
    try:
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n<1:
            n=0
            while cap.grab():n+=1
        if n<1:raise RuntimeError('Zero source frames '+str(path))
        return n
    finally:cap.release()

def hardware_test(path, max_frames=32):
    import torch
    nv=older()
    images, decoder, decoded, n=nv.decode_gpu(path,max_frames)
    torch.cuda.synchronize()
    hw=tuple(images.shape[-2:])
    del images,decoder,decoded
    return int(n),hw

def test_clips(args):
    # Original test index and class are authoritative; class label is 0/1.
    source,_,_,_=parent().load_source('x3d_s',args)
    original=source.scan(args.data,'test')
    if len(original)!=232:raise RuntimeError(f'Expected 232 clips, found {len(original)}')
    counts={0:0,1:0}
    for _,label in original:
        if label not in counts:raise RuntimeError('Unexpected label '+str(label))
        counts[label]+=1
    if counts!={0:116,1:116}:
        raise RuntimeError(f'Expected 116 each (normal/anomalous), got {counts}')
    return original

def final_manifest_path(args):return args.results/'full232_manifest.json'

def manifest_check(args, require_all=True):
    f=final_manifest_path(args)
    if not f.is_file():raise RuntimeError(f'Missing {f}; run --stage prepare first')
    data=json.loads(f.read_text())
    if data.get('manifest_version')!=MANIFEST_VERSION or data.get('data')!=str(args.data):
        raise RuntimeError('Manifest version/source path mismatch; run prepare in a NEW results directory')
    if len(data.get('clips',[]))!=232:raise RuntimeError('Manifest does not contain 232 clips')
    actual=test_clips(args)
    for idx,(rec,(src,label)) in enumerate(zip(data['clips'],actual)):
        if rec['index']!=idx or rec['label']!=label or rec['original']!=str(Path(src).resolve()):
            raise RuntimeError(f'Source manifest drift at row {idx}')
        if not Path(rec['effective']).is_file():
            raise FileNotFoundError(f'Prepared clip is missing: {rec["effective"]}')
        if rec['original_bytes']!=Path(rec['original']).stat().st_size:
            raise RuntimeError('Original clip changed: '+rec['original'])
        if rec['effective_bytes']!=Path(rec['effective']).stat().st_size:
            raise RuntimeError('Prepared clip changed: '+rec['effective'])
    return data

def transcode_h264(original, dest, crf):
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.stem+'.building.mp4')
    if tmp.exists():tmp.unlink()
    # No resize and no change to intended frame rate; conversion is an OFFLINE preparation step.
    cmd=['ffmpeg','-hide_banner','-nostdin','-loglevel','error','-y','-i',str(original),
         '-map','0:v:0','-an','-sn','-dn','-c:v','libx264','-preset','medium',
         '-crf',str(crf),'-pix_fmt','yuv420p','-fps_mode','passthrough',
         '-movflags','+faststart',str(tmp)]
    try:
        run_process(cmd,'H.264 transcode')
        if not tmp.is_file() or tmp.stat().st_size<=0:
            raise RuntimeError('Transcoder emitted empty file')
        os.replace(tmp,dest)
    finally:
        if tmp.exists():tmp.unlink()

def prepare(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('NVIDIA GPU required')
    f=final_manifest_path(args)
    if f.exists():
        found=manifest_check(args)
        print('EXISTING VALID MANIFEST',f,'232 clips; transcoded',found['transcoded_count'],flush=True)
        return found
    original=test_clips(args)
    clips=[]
    transcode_dir=args.results/'h264_prepared'
    transcode_dir.mkdir(exist_ok=True,parents=True)
    count_convert=0
    for i,(src,label) in enumerate(original):
        src=Path(src).resolve()
        before=ffprobe_video(src)
        source_count=opencv_framecount(src)
        mode='original_nvdec'
        try:
            decoded_count,_=hardware_test(src)
            if decoded_count!=source_count:
                raise RuntimeError(f'Frame count mismatch original OpenCV={source_count}, NVDEC={decoded_count}')
            effective=src
        except Exception as err:
            # Capture incompatibility in manifest, never silently skip the video.
            digest=hashlib.sha256(str(src).encode()).hexdigest()[:12]
            effective=transcode_dir/f'{i:03d}_{digest}.mp4'
            if not effective.exists():
                print(f'TRANSCODE {i+1}/232 {src.name}: {str(err)[:130]}',flush=True)
                transcode_h264(src,effective,args.crf)
            after=ffprobe_video(effective)
            if (before['width'],before['height'])!=(after['width'],after['height']):
                raise RuntimeError(f'Resolution changed during transcode for {src}')
            converted_count=opencv_framecount(effective)
            decoded_count,_=hardware_test(effective)
            if converted_count!=source_count or decoded_count!=source_count:
                raise RuntimeError(f'Frame-count mismatch in {src}: original OpenCV {source_count}, '
                    f'converted OpenCV {converted_count}, NVDEC {decoded_count}. '
                    'Please inspect this video; no automatic frame dropping is permitted.')
            mode='transcoded_h264'
            count_convert+=1
        rec={'index':i,'label':int(label),'original':str(src),'effective':str(effective),
             'processing':mode,'original_codec':before['codec'],
             'original_resolution':[before['width'],before['height']],
             'original_fps':before['fps'],'original_frames':source_count,
             'nvdec_frames':decoded_count,
             'original_bytes':src.stat().st_size,'effective_bytes':effective.stat().st_size}
        clips.append(rec)
        if (i+1)%20==0 or i+1==len(original):
            print('PREPARED',i+1,'/',len(original),'transcoded',count_convert,flush=True)
    result={'manifest_version':MANIFEST_VERSION,'data':str(args.data),
            'device':torch.cuda.get_device_name(0),'precision':'BF16 when supported',
            'n_test':len(clips),'n_normal':116,'n_anomalous':116,
            'transcoded_count':count_convert,'native_count':232-count_convert,
            'compatibility_policy':'NVDEC direct, otherwise offline H.264 CRF '+str(args.crf)+
             ' no spatial resizing; frame counts must match OpenCV original',
            'offline_transcode_excluded_from_timed_e2e':True,
            'clips':clips}
    atomic_json(f,result)
    print('MANIFEST READY',f,'transcoded',count_convert,'native',232-count_convert,flush=True)
    return manifest_check(args)

def metric_summary(arr):return older().summary(arr)

def compute_metrics(y,preds):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
    if len(y)!=232 or len(preds)!=232:raise RuntimeError('Wrong number of predictions')
    return {'accuracy_pct':round(100*accuracy_score(y,preds),4),
            'f1_pct':round(100*f1_score(y,preds,zero_division=0),4),
            'precision_pct':round(100*precision_score(y,preds,zero_division=0),4),
            'recall_pct':round(100*recall_score(y,preds,zero_division=0),4),
            'confusion_matrix':confusion_matrix(y,preds,labels=[0,1]).tolist(),
            'correct':sum(a==b for a,b in zip(y,preds)), 'n':len(y)}

def opencv_reference_prediction(source,source_path,cfg,model,amp_dtype):
    import numpy as np
    import torch
    # Inference is equivalent to original evaluation (decode_one + forward_batch).
    rgb=source.decode_one(source_path,int(cfg['frames']),int(cfg['width']),int(cfg['height']))
    cpu=torch.from_numpy(np.ascontiguousarray(rgb)).permute(0,3,1,2).unsqueeze(0).contiguous()
    with torch.inference_mode(),torch.autocast('cuda',dtype=amp_dtype):
        logits,_=source.forward_batch(model,cpu,torch.zeros(1,dtype=torch.long),cfg,False)
    pred=int(logits.float().argmax(-1).item())
    torch.cuda.synchronize()
    del cpu,logits
    return pred

def model_run(args):
    import torch
    name=args.single
    pp=parent(); nv=older()
    records=manifest_check(args)['clips']
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU')
    torch.set_num_threads(min(2,os.cpu_count() or 2))
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print('MODEL',name,'| reference_accuracy',not args.skip_reference,flush=True)
    ckptpath=pp.discover(name,args)
    source,inner,cfg,source_dir=pp.load_source(name,args)
    torch.manual_seed(42)
    model,_=source.build_model(inner,source_dir)
    ckpt=torch.load(ckptpath,map_location='cpu',weights_only=False)
    pp.validate_checkpoint_identity(name,ckpt,cfg)
    model.load_state_dict(pp.state_of(ckpt),strict=True)
    model=model.cuda().eval()
    mean=torch.tensor(cfg['mean'],device='cuda',dtype=torch.float32).view(1,1,3,1,1)
    std=torch.tensor(cfg['std'],device='cuda',dtype=torch.float32).view(1,1,3,1,1)
    torch.cuda.reset_peak_memory_stats()
    print(f'CHECKPOINT VERIFIED {ckptpath} | {cfg["frames"]} frames '
          f'{cfg["width"]}x{cfg["height"]} | {str(amp_dtype)}',flush=True)
    for i in range(args.warmup):
        rec=records[i%232]
        nv.process_one(rec['effective'],cfg,model,amp_dtype,mean,std)
    print('WARMUP COMPLETE',args.warmup,flush=True)
    # IMPORTANT: no stage synchronization in the PRIMARY reported E2E times.
    raw_rows=[]
    y=[r['label'] for r in records]
    predictions=[]
    for pass_num in range(1,args.passes+1):
        preds_this=[];times=[]
        # Same source order for every model; optionally reverse alternating passes
        # to reduce temporal drift effects while preserving 232 unique clips/pass.
        order=list(range(232)) if pass_num%2 else list(reversed(range(232)))
        for step,i in enumerate(order):
            rec=records[i]
            v=nv.process_one(rec['effective'],cfg,model,amp_dtype,mean,std,detail=False)
            times.append(v['e2e_ms']);preds_this.append((i,v['pred']))
            raw_rows.append({'model':name,'pass':pass_num,'index':i,'label':rec['label'],
                   'prepared_type':rec['processing'],'video':rec['original'],
                   'effective_video':rec['effective'], 'e2e_ms':round(v['e2e_ms'],4),
                   'nvdec_pred':v['pred']})
            if (step+1)%58==0:
                print(f'PASS {pass_num}/{args.passes} {step+1}/232',flush=True)
        pred_map=dict(preds_this)
        ordered=[pred_map[i] for i in range(232)]
        predictions.append(ordered)
        print(f'PASS {pass_num} E2E:',metric_summary(times),
              'NVDEC ACCURACY:',compute_metrics(y,ordered)['accuracy_pct'],flush=True)
    if any(p!=predictions[0] for p in predictions[1:]):
        raise RuntimeError('Predictions differed between identical evaluation passes; investigate non-determinism')
    primary_times=[r['e2e_ms'] for r in raw_rows]
    stage_rows=[]
    reference_preds=[]; diag_preds=[]
    for i,rec in enumerate(records):
        # Diagnostics: stage boundaries are synchronized, unlike the primary E2E run.
        d=nv.process_one(rec['effective'],cfg,model,amp_dtype,mean,std,detail=True)
        diag_preds.append(d['pred'])
        row={'index':i,'label':rec['label'],'prepared_type':rec['processing'],
             'decode_open_ms':round(d['decode_ms'],4),
             'gpu_preprocess_ms':round(d['gpu_preprocess_ms'],4),
             'inference_output_ms':round(d['inference_output_ms'],4),
             'diagnostic_e2e_ms':round(d['e2e_ms'],4),
             'nvdec_pred':d['pred']}
        if not args.skip_reference:
            ref=opencv_reference_prediction(source,rec['original'],cfg,model,amp_dtype)
            reference_preds.append(ref)
            row['opencv_original_pred']=ref
            row['prediction_match']=int(ref==d['pred'])
        stage_rows.append(row)
        if (i+1)%58==0:print('DIAGNOSTICS',i+1,'/232',flush=True)
    if diag_preds!=predictions[0]:
        raise RuntimeError('Diagnostic pass predictions differ from primary NVDEC pass')
    summary_obj={'model':name,'checkpoint':str(ckptpath),
         'device':torch.cuda.get_device_name(0),'precision':str(amp_dtype).replace('torch.',''),
         'batch_size':1,'source_test_clips':232,'timing_passes':args.passes,
         'measured_primary_observations':232*args.passes,
         'unique_clips':232,'warmup':args.warmup,'frames':cfg['frames'],
         'input_width':cfg['width'],'input_height':cfg['height'],
         'parameters_m':round(sum(p.numel() for p in model.parameters())/1e6,4),
         'benchmark_definition':'Cold-clip, indexed PyNvVideoCodec NVDEC GPU RGBP, GPU letterbox/normalization, model+argmax; offline compatibility transcode excluded',
         'transcoded_count':sum(r['processing']=='transcoded_h264' for r in records),
         'natively_nvdec_count':sum(r['processing']=='original_nvdec' for r in records),
         'primary_e2e':metric_summary(primary_times),
         'primary_e2e_per_pass':[metric_summary([r['e2e_ms'] for r in raw_rows if r['pass']==p])
                                  for p in range(1,args.passes+1)],
         'stage_diagnostic':{
           'decode_and_open':metric_summary([r['decode_open_ms'] for r in stage_rows]),
           'gpu_preprocess':metric_summary([r['gpu_preprocess_ms'] for r in stage_rows]),
           'inference_and_output':metric_summary([r['inference_output_ms'] for r in stage_rows]),
           'e2e_with_stage_sync':metric_summary([r['diagnostic_e2e_ms'] for r in stage_rows])},
         'nvdec_accuracy':compute_metrics(y,predictions[0]),
         'opencv_original_accuracy':compute_metrics(y,reference_preds) if reference_preds else None,
         'historical_original_accuracy_pct':HISTORICAL_TEST_ACCURACY_PCT.get(name),
         'historical_reference_consistent':(
             abs(compute_metrics(y,reference_preds)['accuracy_pct'] -
                 HISTORICAL_TEST_ACCURACY_PCT[name]) < 0.015) if reference_preds else None,
         'opencv_nvdec_agreement_pct':round(100*sum(a==b for a,b in zip(reference_preds,predictions[0]))/232,4) if reference_preds else None,
         'disagreements':sum(a!=b for a,b in zip(reference_preds,predictions[0])) if reference_preds else None,
         'peak_gpu_allocated_mib':round(torch.cuda.max_memory_allocated()/1024**2,2),
         'meaning':'Accuracy is one checkpoint on one fixed 232-clip test split, not five-seed stability. Three timing passes are repetitions of identical videos.'}
    stem=args.results/name
    def write_csv(path,rows):
        with Path(path).open('w',newline='') as fh:
            wr=csv.DictWriter(fh,fieldnames=list(rows[0]))
            wr.writeheader();wr.writerows(rows)
    write_csv(args.results/f'{name}_timings.csv',raw_rows)
    write_csv(args.results/f'{name}_stages_accuracy.csv',stage_rows)
    # Write the completion marker last, to preserve safe resume behavior.
    atomic_json(stem.with_suffix('.json'),summary_obj)
    if summary_obj['historical_reference_consistent'] is False:
        print('WARNING Original OpenCV accuracy differs from previously recorded test accuracy; investigate before publication.',flush=True)
    if summary_obj['disagreements']:
        print(f'WARNING {summary_obj["disagreements"]} / 232 NVDEC-vs-OpenCV prediction disagreements; report NVDEC accuracy separately.',flush=True)
    print('FINAL RESULT',name,
          'E2E',summary_obj['primary_e2e'],
          'NVDEC accuracy',summary_obj['nvdec_accuracy']['accuracy_pct'],
          'Original OpenCV accuracy', (summary_obj['opencv_original_accuracy'] or {}).get('accuracy_pct'),
          'agreement',summary_obj['opencv_nvdec_agreement_pct'],flush=True)
    del model,ckpt,source
    gc.collect();torch.cuda.empty_cache()
    return summary_obj

def consolidated_csv(args):
    models=list(parent().SOURCES)
    output=[]
    for model in models:
        p=args.results/f'{model}.json'
        if not p.is_file():continue
        d=json.loads(p.read_text())
        fields={'model':model,'parameters_m':d['parameters_m'],
                'frames':d['frames'],'width':d['input_width'],'height':d['input_height'],
                'e2e_median_ms':d['primary_e2e']['median_ms'],
                'e2e_mean_ms':d['primary_e2e']['mean_ms'],
                'e2e_p95_ms':d['primary_e2e']['p95_ms'],
                'e2e_min_ms':d['primary_e2e']['min_ms'],
                'e2e_max_ms':d['primary_e2e']['max_ms'],
                'e2e_pass1_median_ms':d['primary_e2e_per_pass'][0]['median_ms'],
                'e2e_pass2_median_ms':d['primary_e2e_per_pass'][1]['median_ms'],
                'e2e_pass3_median_ms':d['primary_e2e_per_pass'][2]['median_ms'],
                'decode_open_median_ms':d['stage_diagnostic']['decode_and_open']['median_ms'],
                'preprocess_median_ms':d['stage_diagnostic']['gpu_preprocess']['median_ms'],
                'inference_output_median_ms':d['stage_diagnostic']['inference_and_output']['median_ms'],
                'nvdec_accuracy_pct':d['nvdec_accuracy']['accuracy_pct'],
                'nvdec_f1_pct':d['nvdec_accuracy']['f1_pct'],
                'original_opencv_accuracy_pct':(d['opencv_original_accuracy'] or {}).get('accuracy_pct'),
                'historical_accuracy_match':d.get('historical_reference_consistent'),
                'prediction_agreement_pct':d['opencv_nvdec_agreement_pct'],
                'disagreements':d['disagreements'],
                'prepared_h264_clips':d['transcoded_count'],
                'peak_gpu_allocated_mib':d['peak_gpu_allocated_mib']}
        output.append(fields)
    if output:
        out=args.results/'scad17_nvdec_final232_summary.csv'
        with out.open('w',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=list(output[0]))
            w.writeheader();w.writerows(output)
        print('SUMMARY',out,'MODELS',len(output),flush=True)

def run_profile(args):
    manifest_check(args)
    failed=[]
    for name in parent().SOURCES:
        outfile=args.results/(name+'.json')
        if outfile.exists():
            print('SKIP COMPLETED',name,flush=True)
            continue
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage','single',
             '--single',name,'--home',str(args.home),'--data',str(args.data),
             '--checkpoints',str(args.checkpoints),'--results',str(args.results),
             '--passes',str(args.passes),'--warmup',str(args.warmup),
             '--crf',str(args.crf)]
        if args.skip_reference:cmd.append('--skip-reference')
        print('\nRUN MODEL',name,flush=True)
        with (args.results/(name+'_run.log')).open('w') as fh:
            pr=subprocess.Popen(cmd,cwd=args.home,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in pr.stdout:
                sys.stdout.write(line);fh.write(line)
            ret=pr.wait()
        if ret!=0:
            failed.append(name)
            print('FAILED',name,'see',args.results/(name+'_run.log'),flush=True)
        consolidated_csv(args)
    if failed:
        raise RuntimeError('Models failed: '+', '.join(failed))
    if sum((args.results/f'{m}.json').exists() for m in parent().SOURCES)!=17:
        raise RuntimeError('Incomplete outputs; expected 17 models')
    print('COMPLETE ALL 17',flush=True)

def main():
    ap=argparse.ArgumentParser(description='SCAD: final NVDEC cold-clip 17x232, 3-pass benchmark')
    ap.add_argument('--stage',required=True,choices=['prepare','audit','single','profile','summary'])
    ap.add_argument('--single',choices=list(parent().SOURCES))
    ap.add_argument('--home',type=Path,default=HOME)
    ap.add_argument('--data',type=Path)
    ap.add_argument('--checkpoints',type=Path)
    ap.add_argument('--results',type=Path)
    ap.add_argument('--passes',type=int,default=3)
    ap.add_argument('--warmup',type=int,default=15)
    ap.add_argument('--crf',type=int,default=14)
    ap.add_argument('--skip-reference',action='store_true',help='Not recommended: omit original OpenCV accuracy comparison')
    args=ap.parse_args()
    args.home=args.home.expanduser().resolve()
    args.data=(args.data or args.home/'accident').expanduser().resolve()
    args.checkpoints=(args.checkpoints or args.home/'scad'/'models').expanduser().resolve()
    args.results=(args.results or args.home/'scad'/'profiling_17'/'nvdec_final232').expanduser().resolve()
    args.results.mkdir(parents=True,exist_ok=True)
    if args.passes!=3:ap.error('Final protocol fixes exactly 3 timing passes')
    if args.crf<0 or args.crf>25:ap.error('crf must be 0..25')
    if args.warmup<0:ap.error('warmup must be >=0')
    if args.stage=='single' and args.single is None:ap.error('Need --single MODEL')
    if args.stage=='prepare':prepare(args)
    elif args.stage=='audit':
        print('CHECKPOINTS:')
        parent().audit(args)
        data=manifest_check(args)
        print('MANIFEST valid: 232 clips; transcoded',data['transcoded_count'])
    elif args.stage=='single':model_run(args);consolidated_csv(args)
    elif args.stage=='profile':run_profile(args)
    elif args.stage=='summary':consolidated_csv(args)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('FINAL NVDEC BENCHMARK FAILED',str(exc),file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.exit(2)
