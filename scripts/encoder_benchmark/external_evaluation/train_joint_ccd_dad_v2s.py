#!/usr/bin/env python3
"""Joint CCD+DAD VideoMAE V2-S, from SCAD K710-distilled pretrained weights.
Requires the previously provided train_ccd_videomaev2_s_ram.py,
train_ccd_videomae_s_ram.py, train_dad_frontier_fast_ram.py beside this file.
Never trains or selects checkpoints on either held-out test set.
"""
import argparse, csv, gc, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import train_ccd_videomaev2_s_ram as ccd
import train_dad_frontier_fast_ram as dad

class JointData(Dataset):
    def __init__(self, ccd_ds, dad_ds):
        self.a, self.b = ccd_ds, dad_ds
        self.n_a=len(ccd_ds)
    def __len__(self): return self.n_a + len(self.b)
    def __getitem__(self,i):
        return self.a[i] if i<self.n_a else self.b[i-self.n_a]

class BalancedBatchSampler(Sampler):
    """Equal count from each dataset per batch, sampled with replacement.
    Within each dataset the observed class frequencies remain unchanged in expectation.
    Deterministically vary sampling each epoch using (seed,epoch).
    """
    def __init__(self,n_ccd,n_dad,batch,epoch_size,seed):
        if batch%2: raise ValueError('Physical batch must be even')
        self.n_ccd=n_ccd;self.n_dad=n_dad;self.batch=batch
        self.steps=math.ceil(epoch_size/batch);self.seed=seed;self.epoch=0
    def __len__(self):return self.steps
    def __iter__(self):
        rng=np.random.default_rng(self.seed+1009*self.epoch)
        half=self.batch//2
        for _ in range(self.steps):
            a=rng.integers(0,self.n_ccd,size=half).tolist()
            b=(rng.integers(0,self.n_dad,size=half)+self.n_ccd).tolist()
            batch=a+b;rng.shuffle(batch)
            yield batch

def loader(ds,batch=None,sampler=None,workers=4):
    kw=dict(num_workers=workers,pin_memory=True,persistent_workers=workers>0)
    if workers:kw.update(prefetch_factor=2,multiprocessing_context='fork')
    if sampler is not None:return DataLoader(ds,batch_sampler=sampler,**kw)
    return DataLoader(ds,batch_size=batch,shuffle=False,**kw)

def metrics(model,ds,cfg,dtype,batch,workers,with_preds=False):
    data=loader(ds,batch=batch,workers=workers)
    result=ccd.ccd_evaluate(model,data,cfg,dtype,with_predictions=with_preds)
    del data
    return result

def export_preds(path, items, labels, scores, predictions):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','video','label','accident_probability','predicted_label'])
        for item,y,p,pred in zip(items,labels,scores,predictions):
            w.writerow([item[0],item[1],y,f'{p:.10f}',pred])

def main():
    p=argparse.ArgumentParser(description='Dataset-balanced joint fine-tuning on CCD and DAD')
    p.add_argument('--root',type=Path,default=Path('/home/user/22104053'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--effective-batch',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--dry-check',action='store_true',help='Verify splits and cache paths, no training')
    p.add_argument('--rerun',action='store_true')
    args=p.parse_args()
    if not torch.cuda.is_available() and not args.dry_check:raise RuntimeError('CUDA GPU required')
    if args.batch<2 or args.batch%2:raise ValueError('--batch must be an even integer')
    if args.effective_batch%args.batch:raise ValueError('--effective-batch must be divisible by --batch')
    root=args.root;ccd_root=root/'CCD';dad_root=root/'DAD'
    out=(args.out or root/'joint_ccd_dad'/'videomaev2_s'/f'seed_{args.seed}')
    if (out/'test_metrics.json').exists() and not args.rerun:
        print('ALREADY COMPLETE:',out/'test_metrics.json');return
    ccd_rows={};ccd_manifests={}
    for s in ('train','val','test'):
        ccd_rows[s],ccd_manifests[s]=ccd.ccd_entries(ccd_root,s)
    dad_rows={s:dad.read_dad(dad_root,s) for s in ('train','val','test')}
    expected_ccd={'train':3150,'val':450,'test':900}
    expected_dad={'train':1109,'val':175,'test':466}
    for s in expected_ccd:
        assert len(ccd_rows[s])==expected_ccd[s],f'CCD {s}: {len(ccd_rows[s])}'
        assert len(dad_rows[s])==expected_dad[s],f'DAD {s}: {len(dad_rows[s])}'
    for name,groups in [('CCD',ccd_rows),('DAD',dad_rows)]:
        paths=[{str(z[2]) if name=='CCD' else z[0] for z in groups[s]} for s in ('train','val','test')]
        assert not(paths[0]&paths[1] or paths[0]&paths[2] or paths[1]&paths[2]),f'{name} file overlap'
    counts={'CCD':{s:{'total':len(ccd_rows[s]),'accidents':sum(x[1] for x in ccd_rows[s])} for s in ccd_rows},
            'DAD':{s:{'total':len(dad_rows[s]),'accidents':sum(x[1] for x in dad_rows[s])} for s in dad_rows}}
    print('SPLIT CHECK:',json.dumps(counts),flush=True)
    if args.dry_check:
        print('Dry check passed. No video decoding or model construction.');return
    out.mkdir(parents=True,exist_ok=True)
    ccd.seed_all(args.seed)
    cfg=dict(ccd.SPECS['videomaev2_s']);cfg['lr']=args.lr;cfg['batch']=args.batch
    cfg['accum']=args.effective_batch//args.batch
    # Distinct per-dataset caches are retained on disk, with train/validation in CPU RAM.
    cache_ccd={s:ccd.ccd_cache(ccd_rows[s],ccd_manifests[s],ccd_root,s) for s in ('train','val')}
    frames_ccd={s:ccd.ccd_load_ram(cache_ccd[s]) for s in ('train','val')}
    cache_dad=root/'DAD'/'cache'/'frontier_ram'/'16f_224x224'
    frames_dad={s:dad.DADRamVideos(dad_rows[s],cfg,cache_dad,s) for s in ('train','val')}
    data_ccd={s:ccd.CCDRamDataset(frames_ccd[s],ccd_rows[s]) for s in ('train','val')}
    print('All CCD and DAD training/validation frame caches RAM-resident',flush=True)
    train_ds=JointData(data_ccd['train'],frames_dad['train'])
    sampler=BalancedBatchSampler(len(data_ccd['train']),len(frames_dad['train']),args.batch,
                                 len(train_ds),args.seed)
    train_dl=loader(train_ds,sampler=sampler,workers=args.workers)
    val_ccd=data_ccd['val'];val_dad=frames_dad['val']
    model,pretraining=ccd.build_model('videomaev2_s',root/'.cache/scad_variants/sources')
    model=model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=cfg['wd'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.05)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    config={'seed':args.seed,'model':'videomaev2_s','initialization':pretraining,'epochs':args.epochs,
            'lr':args.lr,'weight_decay':cfg['wd'],'physical_batch':args.batch,'effective_batch':args.effective_batch,
            'training_sampling':'50% CCD, 50% DAD per physical batch; replacement; original within-dataset label frequencies in expectation',
            'epoch_draws':len(sampler)*args.batch,'checkpoint_selection':'equal-weight CCD and DAD validation macro-F1; mean validation loss tie break',
            'target_threshold':0.5,'preprocessing':'uniform 16 RGB frames, 224x224 letterbox, GPU normalization',
            'class_mapping':'0=normal, 1=accident','counts':counts}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    best_score=-1;best_loss=float('inf');history=[]
    for ep in range(1,args.epochs+1):
        sampler.epoch=ep
        model.train();optimizer.zero_grad(set_to_none=True)
        t0=time.time();loss_sum=0.;seen=0;steps=len(train_dl)
        for i,(x,y) in enumerate(train_dl):
            block=min(cfg['accum'],steps-(i//cfg['accum'])*cfg['accum'])
            with torch.autocast('cuda',dtype=dtype):
                logits,yy=ccd.forward_batch(model,x,y,cfg,True)
                loss=F.cross_entropy(logits.float(),yy)
            scaler.scale(loss/block).backward()
            if (i+1)%cfg['accum']==0 or i+1==steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            loss_sum+=loss.item()*len(y);seen+=len(y)
        vc=metrics(model,val_ccd,cfg,dtype,args.batch,args.workers)
        vd=metrics(model,val_dad,cfg,dtype,args.batch,args.workers)
        score=.5*(vc['macro_f1']+vd['macro_f1'])
        loss_tie=.5*(vc['loss']+vd['loss'])
        rec={'epoch':ep,'train_loss':loss_sum/seen,'ccd_val_accuracy':vc['accuracy'],
             'ccd_val_macro_f1':vc['macro_f1'],'dad_val_accuracy':vd['accuracy'],
             'dad_val_macro_f1':vd['macro_f1'],'balanced_val_macro_f1':score,
             'balanced_val_loss':loss_tie,'epoch_seconds':round(time.time()-t0,2),
             'lr':optimizer.param_groups[0]['lr']}
        print('EPOCH',json.dumps(rec),flush=True);history.append(rec)
        if score>best_score+1e-12 or (abs(score-best_score)<1e-12 and loss_tie<best_loss):
            best_score=score;best_loss=loss_tie
            torch.save({'model':model.state_dict(),'epoch':ep,'ccd_val':vc,'dad_val':vd,
                        'score':score,'pretraining':pretraining},out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        scheduler.step()
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    # Free training RAM before test caches, critical for 32 GB machines.
    del train_dl,train_ds,data_ccd,frames_ccd,frames_dad,val_ccd,val_dad
    gc.collect()
    summary={'model':'videomaev2_s','seed':args.seed,'best_epoch':checkpoint['epoch'],
             'balanced_best_validation_macro_f1':checkpoint['score'],'datasets':{}}
    for ds in ('CCD','DAD'):
        if ds=='CCD':
            cache=ccd.ccd_cache(ccd_rows['test'],ccd_manifests['test'],ccd_root,'test')
            test_data=ccd.CCDRamDataset(ccd.ccd_load_ram(cache),ccd_rows['test'])
            items=[('CCD',r[0]) for r in ccd_rows['test']]
        else:
            test_data=dad.DADRamVideos(dad_rows['test'],cfg,cache_dad,'test')
            items=[('DAD',str(Path(r[0]).relative_to(dad_root/'videos'))) for r in dad_rows['test']]
        result,ys,probs,preds=metrics(model,test_data,cfg,dtype,args.batch,args.workers,True)
        summary['datasets'][ds]=result
        export_preds(out/f'{ds.lower()}_test_predictions.csv',items,ys,probs,preds)
        print(f'FINAL TEST {ds}:',json.dumps(result),flush=True)
        del test_data;gc.collect()
    (out/'test_metrics.json').write_text(json.dumps(summary,indent=2))
    print('FINISHED',out/'test_metrics.json',flush=True)

if __name__=='__main__':main()
