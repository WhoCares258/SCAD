#!/usr/bin/env python3
"""SCAD manuscript evidence audit. READ-ONLY: no training, no GPU, no inference.

Reads existing NVDEC 50-checkpoint final batch-1 predictions, timings and
training metadata; exports all classwise and diagnostic manuscript tables.

python scad_extract_evidence.py
"""
from __future__ import annotations
import argparse, csv, json, math, statistics, zipfile
from pathlib import Path
from collections import defaultdict, Counter

MODELS = [
 'vjepa21_l','videomae_v1_l','videomaev2_b','vjepa21_b_full',
 'videomaev2_s','internvideo2_l','mvitv2_b','internvideo2_b',
 'mvitv2_s','internvideo2_s']
SEEDS=(42,43,44,45,46)
PAIRS=[('vjepa21_l','videomae_v1_l'),('videomae_v1_l','videomaev2_s'),
       ('videomaev2_s','internvideo2_s'),('vjepa21_l','videomaev2_s')]
N=232

def readcsv(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def writecsv(p, rows, fields=None):
 rows=list(rows)
 if not rows:return
 if fields is None:fields=list(rows[0].keys())
 with p.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
def div(a,b):return a/b if b else 0.
def pct(a,b):return 100*div(a,b)
def fmean(vals):return statistics.mean(vals) if vals else float('nan')
def fsd(vals):return statistics.stdev(vals) if len(vals)>1 else 0.
def pctile(x,p):
 x=sorted(x); q=(len(x)-1)*p/100; i=int(q);j=min(i+1,len(x)-1)
 return x[i]+(x[j]-x[i])*(q-i)
def cm(y,p):
 tn=sum(a==0 and b==0 for a,b in zip(y,p))
 fp=sum(a==0 and b==1 for a,b in zip(y,p))
 fn=sum(a==1 and b==0 for a,b in zip(y,p))
 tp=sum(a==1 and b==1 for a,b in zip(y,p))
 prec1=div(tp,tp+fp);rec1=div(tp,tp+fn)
 prec0=div(tn,tn+fn);rec0=div(tn,tn+fp)
 f1_1=div(2*prec1*rec1,prec1+rec1)
 f1_0=div(2*prec0*rec0,prec0+rec0)
 return dict(tn=tn,fp=fp,fn=fn,tp=tp,n=len(y),correct=tn+tp,
    accuracy_pct=pct(tn+tp,len(y)),macro_f1_pct=50*(f1_0+f1_1),
    accident_precision_pct=100*prec1,accident_recall_pct=100*rec1,
    accident_f1_pct=100*f1_1,normal_precision_pct=100*prec0,
    normal_recall_pct=100*rec0,normal_f1_pct=100*f1_0,
    balanced_accuracy_pct=50*(rec0+rec1),
    fpr_pct=pct(fp,fp+tn),fnr_pct=pct(fn,fn+tp))
def rdjson(p):return json.loads(p.read_text())
def nice(x):return x.replace('_',' ')
def main():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument('--home',type=Path,default=Path('/home/user/22104053'))
 a.add_argument('--benchmark',type=Path,default=None)
 a.add_argument('--training',type=Path,default=None)
 a.add_argument('--output',type=Path,default=None)
 a.add_argument('--allow-incomplete',action='store_true')
 args=a.parse_args()
 home=args.home.expanduser().resolve()
 bench=args.benchmark or home/'scad/final50_checkpoint_latency'
 train=args.training or home/'scad/nvdec_final10'
 out=args.output or home/'scad/manuscript_evidence'
 out.mkdir(parents=True,exist_ok=True)
 for p in (bench,train):
  if not p.is_dir():raise SystemExit(f'MISSING DIRECTORY: {p}')

 manifest_path=train/'prepared_manifest.json'
 source_by_index={}
 if manifest_path.is_file():
  m=rdjson(manifest_path)
  for r in m.get('records',[]):
   if r.get('split')=='test':
    idx=int(r['index']);source_by_index[idx]=r
  manifest_audit={
   'split_counts':dict(Counter(r.get('split') for r in m.get('records',[]))),
   'test_labels':dict(Counter(str(r.get('label')) for r in m.get('records',[]) if r.get('split')=='test')),
   'test_processing':dict(Counter(r.get('processing') for r in m.get('records',[]) if r.get('split')=='test')),
   'source_original_paths_present':bool(source_by_index),
   'note':'Prepared-source transcoding occurred before timed inference; excluded from E2E.'}
 else:manifest_audit={'warning':'prepared_manifest.json unavailable'}
 (out/'dataset_manifest_audit.json').write_text(json.dumps(manifest_audit,indent=2))

 per_seed=[]; per_model=[]; label_reference=None; predictions={};clip_errors=[]
 timing_class=[]; timing_type=[];slowest=[]; epoch=[];missing=[]
 summaries=[]
 for model in MODELS:
  modelrows=[];modeltimings=[]; byindex=defaultdict(list)
  pred_all={}
  for seed in SEEDS:
   base=bench/'runs'/model/f'seed_{seed}'
   predfile=base/'predictions.csv'
   complete=base/'complete.json'
   stamp=train/'runs'/model/f'seed_{seed}'/'final10_complete.json'
   if not predfile.is_file() or not complete.is_file() or not stamp.is_file():
    missing.append(f'{model} seed {seed}: missing {predfile if not predfile.is_file() else (complete if not complete.is_file() else stamp)}')
    continue
   meta=rdjson(complete);training=rdjson(stamp)
   pr=readcsv(predfile)
   if len(pr)!=N:raise ValueError(f'{predfile}: expected {N}, found {len(pr)}')
   indexes=[int(r['index']) for r in pr]
   if sorted(indexes)!=list(range(N)):raise ValueError(f'{predfile}: duplicate/missing indexes')
   byidx={int(r['index']):r for r in pr}
   yy=[int(byidx[i]['label']) for i in range(N)]
   pp=[int(byidx[i]['prediction']) for i in range(N)]
   if set(yy)!=set((0,1)) or yy.count(0)!=116 or yy.count(1)!=116:raise ValueError(f'{predfile}: labels not balanced 116/116')
   if label_reference is None:label_reference=yy
   if yy!=label_reference:raise ValueError(f'{predfile}: label/index mismatch relative to other models/seeds')
   if any(v not in (0,1) for v in pp):raise ValueError(f'{predfile}: unexpected prediction label')
   mm=cm(yy,pp)
   if not math.isclose(mm['accuracy_pct'],float(meta['accuracy_pct']),abs_tol=1e-5):
    raise ValueError(f'{model} seed {seed}: computed accuracy differs from benchmark')
   if not math.isclose(mm['macro_f1_pct'],float(meta['macro_f1_pct']),abs_tol=1e-5):
    raise ValueError(f'{model} seed {seed}: computed F1 differs from benchmark')
   detail=dict(model=model,seed=seed,**mm,
        training_accuracy_pct=float(training.get('test_accuracy_pct',float('nan'))),
        batch1_delta_pp=float(meta.get('batch1_minus_training_pp',float('nan'))),
        e2e_mean_ms=float(meta['e2e_mean_ms']),
        e2e_median_ms=float(meta['e2e_median_ms']),
        e2e_p95_ms=float(meta['e2e_p95_ms']),
        best_epoch=training.get('best_epoch'),
        best_val_accuracy_pct=training.get('validation_accuracy_pct'),
        training_wall_seconds=training.get('training_wall_seconds'))
   per_seed.append(detail);modelrows.append(detail)
   pred_all[seed]=pp
   epoch.append({'model':model,'seed':seed,
       'best_epoch':training.get('best_epoch'),
       'best_val_accuracy_pct':training.get('validation_accuracy_pct'),
       'batch1_test_accuracy_pct':mm['accuracy_pct'],
       'test_minus_best_val_pp':mm['accuracy_pct']-float(training.get('validation_accuracy_pct',0)),
       'training_wall_seconds':training.get('training_wall_seconds')})
   timingfile=base/'timings.csv'
   if timingfile.is_file():
    t=readcsv(timingfile)
    if len(t)!=696:raise ValueError(f'{timingfile}: expected 696 timing samples')
    for r in t:
     idx=int(r['index']);v=float(r['e2e_ms']);lab=int(r['label'])
     if lab!=label_reference[idx]:raise ValueError(f'{timingfile}: label mismatch')
     if not math.isfinite(v) or v<=0:raise ValueError(f'{timingfile}: invalid timing')
     byindex[idx].append(v)
     modeltimings.append((lab,r.get('prepared_type','unknown'),v))
   else:missing.append(f'MISSING timings (optional): {timingfile}')
  predictions[model]=pred_all
  if len(modelrows)!=5:
   if not args.allow_incomplete:raise ValueError(f'{model}: only {len(modelrows)} of 5 checkpoint files found. '+'; '.join(missing[-5:]))
   continue
  d={'model':model,'n_seeds':len(modelrows)}
  for metric in ('accuracy_pct','macro_f1_pct','accident_recall_pct','normal_recall_pct',
      'accident_precision_pct','normal_precision_pct','balanced_accuracy_pct',
      'fp','fn','tp','tn','e2e_mean_ms','e2e_median_ms','e2e_p95_ms','batch1_delta_pp'):
   vals=[r[metric] for r in modelrows]
   d[f'mean_{metric}']=fmean(vals)
   d[f'sd_{metric}']=fsd(vals)
  d.update(pooled_tp=sum(r['tp'] for r in modelrows),pooled_tn=sum(r['tn'] for r in modelrows),
       pooled_fp=sum(r['fp'] for r in modelrows),pooled_fn=sum(r['fn'] for r in modelrows),
       note='Pooled counts are 5 repeated predictions for same 232 clips, not 1160 independent clips')
  for j in range(N):
   wrong=[s for s in SEEDS if pred_all[s][j]!=label_reference[j]]
   src=source_by_index.get(j,{})
   clip_errors.append({'model':model,'clip_index':j,'label':label_reference[j],
          'errors_out_of_5':len(wrong),'error_seeds':','.join(map(str,wrong)),
          'filename':Path(src.get('original','')).name if src.get('original') else '',
          'original_video_path':src.get('original','')})
  d['clips_wrong_all_5']=sum(r['model']==model and r['errors_out_of_5']==5 for r in clip_errors)
  d['clips_wrong_3plus']=sum(r['model']==model and r['errors_out_of_5']>=3 for r in clip_errors)
  per_model.append(d)
  if modeltimings:
   for label in (0,1):
    sub=[v for lab,kind,v in modeltimings if lab==label]
    timing_class.append({'model':model,'label':label,'n_timing_samples':len(sub),
      'mean_e2e_ms':fmean(sub),'p95_e2e_ms':pctile(sub,95)})
   for kind in sorted(set(k for lab,k,v in modeltimings)):
    sub=[v for lab,k,v in modeltimings if k==kind]
    timing_type.append({'model':model,'preparation_type':kind,'n_timing_samples':len(sub),
       'mean_e2e_ms':fmean(sub),'p95_e2e_ms':pctile(sub,95)})
   for idx,values in byindex.items():
    src=source_by_index.get(idx,{})
    slowest.append({'model':model,'clip_index':idx,'label':label_reference[idx],
      'n_timing_samples':len(values),'clip_mean_e2e_ms':fmean(values),
      'clip_p95_e2e_ms':pctile(values,95),
      'prepared_type':src.get('processing',''),
      'filename':Path(src.get('original','')).name if src.get('original') else '',
      'original_video_path':src.get('original','')})

 if missing and not args.allow_incomplete:raise ValueError('\n'.join(missing[:12]))
 if not per_model:raise ValueError('No complete five-seed results')
 per_model.sort(key=lambda r:-r['mean_accuracy_pct'])
 writecsv(out/'01_final_model_class_metrics.csv',per_model)
 writecsv(out/'02_per_seed_confusion_and_class_metrics.csv',per_seed)
 writecsv(out/'03_training_validation_test_gap.csv',epoch)
 writecsv(out/'04_per_clip_error_frequency.csv',clip_errors)
 writecsv(out/'05_latency_by_true_class.csv',timing_class)
 writecsv(out/'06_latency_by_transcode_type.csv',timing_type)
 writecsv(out/'07_per_clip_mean_latency.csv',slowest)

 pairwise=[]
 for a,b in PAIRS:
  if a not in predictions or b not in predictions:continue
  matched=[s for s in SEEDS if s in predictions[a] and s in predictions[b]]
  diffs=[];perseed=[]
  for seed in matched:
   pa=predictions[a][seed];pb=predictions[b][seed]
   acert=[pa[i]==label_reference[i] for i in range(N)]
   bcert=[pb[i]==label_reference[i] for i in range(N)]
   Aonly=sum(x and not y for x,y in zip(acert,bcert))
   Bonly=sum(y and not x for x,y in zip(acert,bcert))
   botherr=sum(not x and not y for x,y in zip(acert,bcert))
   bothcorrect=N-Aonly-Bonly-botherr
   diff=100*(Aonly-Bonly)/N;diffs.append(diff)
   pairwise.append(dict(model_A=a,model_B=b,seed=seed,
     accuracy_A_minus_B_pp=diff,A_only_correct=Aonly,B_only_correct=Bonly,
     both_incorrect=botherr,both_correct=bothcorrect))
  if matched:
   summaries.append(dict(model_A=a,model_B=b,paired_seeds=len(matched),
       mean_A_minus_B_pp=fmean(diffs),sd_A_minus_B_pp=fsd(diffs),
       A_wins=sum(v>1e-6 for v in diffs),B_wins=sum(v< -1e-6 for v in diffs),ties=sum(abs(v)<1e-6 for v in diffs)))
 writecsv(out/'08_paired_per_seed_model_comparisons.csv',pairwise)
 writecsv(out/'09_paired_model_comparison_summary.csv',summaries)
 top3={'vjepa21_l','videomae_v1_l','videomaev2_s'}
 hard=[r for r in clip_errors if r['model'] in top3 and r['errors_out_of_5']>=3]
 hard.sort(key=lambda r:(r['model'],-r['errors_out_of_5'],r['clip_index']))
 writecsv(out/'10_review_repeatedly_misclassified_clips.csv',hard)
 tails=[r for r in slowest if r['model'] in top3]
 tails=sorted(tails,key=lambda r:(r['model'],-r['clip_mean_e2e_ms']))
 tops=[]
 for name in sorted(top3):tops.extend([r for r in tails if r['model']==name][:12])
 writecsv(out/'11_review_slowest_clips_top3.csv',tops)

 # Explicit source provenance prompt, never infer dataset family from basename.
 if source_by_index:
  source_rows=[]
  for i in sorted(source_by_index):
   rec=source_by_index[i]
   source_rows.append(dict(clip_index=i,label=rec['label'],
       source_dataset='FILL_ONLY_IF_KNOWN',source_group_id='FILL_ONLY_IF_KNOWN',
       original_video_path=rec.get('original','')))
  writecsv(out/'12_optional_source_provenance_template.csv',source_rows)

 lines=['SCAD evidence collection: 50 saved checkpoints, no training or GPU',
        'Per-class label definitions: 0 = normal, 1 = accident.',
        'Confusion matrix convention: rows=true (normal, accident); cols=pred (normal, accident).',
        'TN = normal predicted normal; FP = normal predicted accident; FN = accident predicted normal; TP = accident predicted accident.',
        'Pooled confusion counts across five seeds reuse the SAME 232 clips; not independent held-out observations.',
        'Mean and SD computed across 5 checkpoints/seeds (not 696 timings treated as independent seeds).',
        '18 offline transcodes are not included in timed E2E; timings relate to prepared file processing.',
        '',
        f'Files read: {sum(len(predictions[m]) for m in predictions)} of 50 checkpoint predictions.',
        '']
 lines.append(f"{'Model':<22} {'Accuracy mean±SD':>18} {'Acc recall':>13} {'Norm recall':>13} {'FN/seed':>9} {'FP/seed':>9} {'E2E ms':>11}")
 for r in per_model:
  lines.append(f"{r['model']:<22} {r['mean_accuracy_pct']:>6.2f} ± {r['sd_accuracy_pct']:<5.2f}"
       f" {r['mean_accident_recall_pct']:>12.2f} {r['mean_normal_recall_pct']:>12.2f}"
       f" {r['mean_fn']:>9.2f} {r['mean_fp']:>9.2f} {r['mean_e2e_mean_ms']:>11.2f}")
 lines+=['','Paired matched-seed model differences (descriptive only; not statistical significance):']
 for r in summaries:
  lines.append(f"{r['model_A']} vs {r['model_B']}: {r['mean_A_minus_B_pp']:+.3f} pp ± {r['sd_A_minus_B_pp']:.3f}; seed wins {r['A_wins']}:{r['B_wins']}, ties {r['ties']}")
 lines+=['','Source-group subgroup results require an authoritative source dataset/group mapping; the optional template is intentionally blank.',
         'ROC AUC, PR AUC and calibration are not recoverable from hard-label B1 predictions. Do not fabricate them.',
         'All detailed data saved to individual CSVs and the ZIP archive.']
 (out/'SCAD_EVIDENCE_README.txt').write_text('\n'.join(lines)+'\n')
 (out/'audit_missing_files.txt').write_text(('\n'.join(missing) if missing else 'All required files found.')+'\n')
 zip_path=out/'scad_manuscript_evidence.zip'
 with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(out.glob('*.csv')):z.write(p,arcname=p.name)
  for p in sorted(out.glob('*.txt')):z.write(p,arcname=p.name)
  z.write(out/'dataset_manifest_audit.json','dataset_manifest_audit.json')
 print('\n'.join(lines))
 print('\nARCHIVE:',zip_path)
 if missing:print('WARNINGS:',len(missing),'see audit_missing_files.txt')

if __name__=='__main__':main()
