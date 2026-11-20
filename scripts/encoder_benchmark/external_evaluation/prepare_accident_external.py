#!/usr/bin/env python3
"""Generate 4s/3s SCAD-style ACCIDENT manifest without extracting/copying video.

Only strict complete windows. Timestamp-derived labels are PROVISIONAL. Windows
with a first-impact timestamp are accident candidates; windows fully preceding
first impact are normal candidates. Post-impact windows are excluded. Source
IDs must be deduplicated against SCAD before publication.
"""
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path

CATS = ('head-on','rear-end','sideswipe','single','t-bone')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.home())
    ap.add_argument('--output',type=Path,default=None)
    ap.add_argument('--verify-files',action='store_true')
    args=ap.parse_args()
    root=args.root.resolve()
    base=root/'real_videos'
    labels=base/'labels.csv'
    if not labels.exists(): raise SystemExit(f'Missing {labels}')
    out=(args.output or (root/'scad'/'external_accident'/'manifests')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    retained=[]; excluded=[]; totals=Counter(); bytype=defaultdict(Counter)
    sources=set(); missing=[]; invalid=[]
    with labels.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        required={'path','type','duration','accident_time'}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit('Missing CSV columns: '+str(required-set(reader.fieldnames or [])))
        for rec in reader:
            rel=rec['path'].strip().replace('\\','/')
            path=(base/rel).resolve()
            source=rel
            if source in sources:
                raise SystemExit(f'Duplicate manifest source {source}')
            sources.add(source)
            try:
                duration=float(rec['duration']); impact=float(rec['accident_time'])
                if not (math.isfinite(duration) and math.isfinite(impact) and duration>=0 and impact>=0):
                    raise ValueError('Invalid duration/impact')
            except (ValueError,TypeError) as e:
                invalid.append((source,str(e))); continue
            if args.verify_files and not path.is_file(): missing.append(str(path))
            cat=rec['type'].strip().lower()
            if cat not in CATS: raise SystemExit(f'Unexpected accident category: {cat} ({source})')
            n=max(0,math.floor((duration-4.0+1e-8)/3.0)+1)
            totals['source_videos']+=1; bytype[cat]['sources']+=1
            for index in range(n):
                start=3.0*index; end=start+4.0
                # Equality at end is pre-impact: first impact not inside [start,end).
                if start <= impact < end:
                    label='accident'; y=1
                elif end <= impact:
                    label='normal_candidate'; y=0
                else:
                    label='post_impact_excluded'; y=None
                row={
                    'source_id':source,'video_path':str(path),'collision_type':cat,
                    'duration_s':f'{duration:.6f}', 'impact_time_s':f'{impact:.6f}',
                    'window_index':index,'start_s':f'{start:.6f}', 'end_s':f'{end:.6f}',
                    'label':label,'label_binary':y if y is not None else '',
                    'original_iid_split':rec.get('split_in_distribution',''),
                    'original_geo_split':rec.get('split_geo_aware',''),
                    'review_status':'UNVERIFIED','scad_overlap_status':'UNCHECKED'
                }
                totals[label]+=1; bytype[cat][label]+=1
                if y is None: excluded.append(row)
                else: retained.append(row)
            if n==0: totals['under_4s_sources']+=1
    fields=['source_id','video_path','collision_type','duration_s','impact_time_s','window_index','start_s','end_s','label','label_binary','original_iid_split','original_geo_split','review_status','scad_overlap_status']
    for name, rows in [('windows_candidates.csv',retained),('windows_postimpact_excluded.csv',excluded)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={
        'input_labels':str(labels),'output':str(out), 'status':'PROVISIONAL: no manual labels or source-overlap audit',
        'counts':dict(totals),'retained_total':len(retained),'excluded_total':len(excluded),
        'by_collision_type':{k:dict(v) for k,v in sorted(bytype.items())},
        'missing_files':missing[:20],'missing_file_count':len(missing), 'invalid_rows':invalid[:20],
        'rules':'4s windows, stride 3s, only full windows; normal iff end<=impact; accident iff start<=impact<end; post if start>impact'
    }
    (out/'manifest_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if len(retained)!=4613 or totals['accident']!=2486 or totals['normal_candidate']!=2127:
        print('WARNING: counts differ from previously calculated strict dataset; inspect source labels and rules.')
    if missing: raise SystemExit(f'{len(missing)} video paths missing')

if __name__=='__main__':main()
