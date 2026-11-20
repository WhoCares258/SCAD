#!/usr/bin/env python3
"""SCAD 17-configuration L40S pipeline.

TRAIN: reproduce four existing single-seed SCAD runs using unmodified original
       runners (10 epochs, seed 42, original checkpoint-selection rule).
AUDIT: inspect one checkpoint per model directory, no filename assumptions.
PROFILE: separately reconstruct every original architecture, strictly load its
       SCAD checkpoint, measure batch-1 inference and actual video end-to-end
       latency, plus PyTorch operator-accounted GFLOPs.

The profiler writes a separate results directory and never changes model files.
GPU execution has to occur on the user's remote L40S; not validated locally.
"""
import argparse
import csv
import gc
import importlib.util
from functools import lru_cache
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

SOURCES = {
    'videomae_v1_l': ('scad_final_nine_l40s.py', 'videomae_v1_l'),
    'vjepa21_l': ('scad_variants_fixed_v2.py', 'vjepa21_l'),
    'vjepa21_b_full': ('scad_next_seven_no_flash_v2.py', 'vjepa21_full'),
    'internvideo2_l': ('scad_variants_fixed_v2.py', 'internvideo2_l'),
    'mvitv2_b': ('scad_variants_fixed_v2.py', 'mvitv2_b'),
    'videomaev2_b': ('scad_six_l40s.py', 'videomaev2_b'),
    'videomaev2_s': ('scad_variants_fixed_v2.py', 'videomaev2_s'),
    'mvitv2_s': ('scad_six_l40s.py', 'mvitv2_s'),
    'videomaev2_l': ('scad_large_two_v3.py', 'videomaev2_l'),
    'videomae_v1_b': ('scad_videomae_v1_sb.py', 'videomae_v1_b'),
    'mvitv2_l_maskfeat16': ('scad_mvitv2_l_maskfeat16_FIXED_v2.py', 'mvitv2_l_maskfeat16'),
    'internvideo2_b': ('scad_next_seven_no_flash_v2.py', 'internvideo2_b'),
    'internvideo2_s': ('scad_final_nine_l40s.py', 'internvideo2_s'),
    'x3d_m': ('scad_final_nine_l40s.py', 'x3d_m'),
    'x3d_s_480x360': ('scad_six_l40s.py', 'x3d_s'),
    'x3d_s': ('scad_next_seven_no_flash_v2.py', 'x3d_s_original'),
    'slowfast_r50': ('scad_variants_fixed_v2.py', 'slowfast_r50'),
}
# Four missing fine-tuned checkpoints. Run exactly the original source scripts.
RETRAIN = ('x3d_s', 'x3d_s_480x360', 'x3d_m', 'slowfast_r50')
DEFAULT_HOME = Path('/home/user/22104053')


def script_for(name, work):
    filename = SOURCES[name][0]
    # Prefer version-matched runners bundled alongside this file.
    base = Path(__file__).resolve().parent
    for path in (base / 'scad_17_sources' / filename, base / filename, work / filename):
        if path.is_file():
            return path
    raise FileNotFoundError('Runner for %s is unavailable: %s. Extract the complete ZIP first.' % (name, filename))


def default_sources(script, home):
    group = ({'scad_six_l40s.py': 'scad_six',
              'scad_next_seven_no_flash_v2.py': 'scad_next',
              'scad_variants_fixed_v2.py': 'scad_variants',
              'scad_final_nine_l40s.py': 'scad_variants',
              'scad_large_two_v3.py': 'scad_variants',
              'scad_videomae_v1_sb.py': 'scad_variants',
              'scad_mvitv2_l_maskfeat16_FIXED_v2.py': 'scad_variants'}[script])
    return home / '.cache' / group / 'sources'


def train_missing(args):
    """Run each original trainer unchanged in its known model folder."""
    for name in RETRAIN:
        path = script_for(name, args.home)
        _, inner = SOURCES[name]
        out = args.checkpoints / name
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-u', str(path), '--models', inner, '--epochs', '10',
               '--seed', '42', '--data', str(args.data), '--out', str(out),
               '--sources', str(default_sources(path.name, args.home)),
               '--workers', str(args.workers)]
        if path.name != 'scad_six_l40s.py':
            cmd.append('--rerun')
        print('\nTRAINING %s\n%s\n' % (name, ' '.join(cmd)), flush=True)
        with (args.results / ('retrain_' + name + '.log')).open('w') as logfile:
            process = subprocess.Popen(cmd, cwd=args.home, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
                logfile.write(line)
            ret = process.wait()
        if ret != 0:
            raise RuntimeError('Training failed: %s. See %s' % (name, args.results / ('retrain_' + name + '.log')))
        print('RETRAIN COMPLETE:', name, flush=True)


@lru_cache(maxsize=8)
def checkpoint_inventory(home_str, checkpoint_root_str):
    """Read-only catalog of existing fine-tuned .pt files, no external downloads.

    Historical runs saved either in ~/scad/models/<name>, ~/scad/incoming,
    ~/scad subdirectories, or directly under the user home. Deliberately do
    not traverse the shared venv, large model caches, or raw dataset folders.
    """
    home = Path(home_str)
    root = Path(checkpoint_root_str)
    roots = [root, home / 'scad', home / 'scad_screen',
             home / 'scad_screen_results']
    found = set()
    for f in home.glob('*.pt'):
        if f.is_file(): found.add(f.resolve())
    for root in roots:
        if not root.is_dir(): continue
        for base, subdirs, names in os.walk(root, followlinks=False):
            subdirs[:] = [n for n in subdirs if n not in
                          {'venv', '.git', '.cache', 'accident', 'sources',
                           'checkpoints', 'scad_17_sources', 'profiling_17'}
                          and not (Path(base)/n).is_symlink()]
            for n in names:
                if n.endswith('.pt'):
                    f = Path(base) / n
                    if f.is_file() and not f.is_symlink():
                        found.add(f.resolve())
    return tuple(sorted(found))


def expected_checkpoint_names(name):
    """Original runner checkpoint basenames plus historical alternate prefixes."""
    inner = SOURCES[name][1]
    variants = {name, inner}
    # For example, full-fine-tuned V-JEPA-B saves as vjepa21_full;
    # the initial X3D-S run saves as x3d_s_original.
    if name == 'x3d_s_480x360': variants = {'x3d_s'}
    if name == 'x3d_s': variants = {'x3d_s_original'}
    if name == 'vjepa21_b_full': variants = {'vjepa21_full', 'vjepa21_b_full'}
    if name == 'mvitv2_l_maskfeat16': variants = {'mvitv2_l_maskfeat16'}
    return {f'{prefix}{variant}_best.pt'
            for variant in variants
            for prefix in ('scad_variant_', 'scad_next_', 'scad6_')}


def discover(name, args):
    """Locate exactly one saved best checkpoint by model directory or filename.

    Prefer canonical folder, then historical model folders, then exact original
    checkpoint basenames. Never select a model merely because its name is a
    substring of another model's filename (e.g. V1-B vs V1-L).
    """
    directory = args.checkpoints / name
    direct = sorted(f.resolve() for f in directory.glob('*.pt') if f.is_file()) if directory.is_dir() else []
    if direct:
        if len(direct) != 1:
            raise RuntimeError(f'{name}: {len(direct)} .pt files in {directory}: {direct}')
        return direct[0]

    paths = checkpoint_inventory(str(args.home), str(args.checkpoints))
    aliases = {name, SOURCES[name][1]}
    if name == 'x3d_s_480x360': aliases = {'x3d_s_480x360'}
    if name == 'x3d_s': aliases = {'x3d_s_original', 'x3d_s'}
    if name == 'vjepa21_b_full': aliases.add('vjepa21_full')
    # The dedicated higher-resolution variant formerly lived under excluded/.
    # Never auto-substitute the 160px X3D-S checkpoint into the 480px run.
    allowed_dirs = {args.home/'scad'/'models'/a for a in aliases}
    allowed_dirs |= {args.home/'scad'/'excluded'/a for a in aliases}
    folder_candidates = [f for f in paths if f.parent in allowed_dirs]
    if len(folder_candidates) == 1:
        return folder_candidates[0]
    if len(folder_candidates) > 1:
        raise RuntimeError(f'{name}: multiple model-folder checkpoints: {folder_candidates}')

    exact_names = expected_checkpoint_names(name)
    candidates = [f for f in paths if f.name in exact_names]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f'{name}: ambiguous exact-name matches: {candidates}. '
                           'Keep one original best.pt or place it in the canonical model folder.')
    # Older, differently named single-model folders: accept only if the folder
    # itself is a unique exact alias, not a substring or a guessed checkpoint.
    candidates = [f for f in paths if f.parent.name in aliases and
                  f.name.lower().endswith('.pt')]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f'{name}: multiple files under alias folders: {candidates}')
    example = sorted(str(f) for f in paths if name.split('_')[0] in str(f).lower())[:5]
    raise RuntimeError(f'{name}: no checkpoint matched across {len(paths)} indexed .pt files. '
                       f'Searched {directory}, ~/scad/**, ~/scad_screen*/**, and ~/*.pt. '
                       f'Potential paths (not auto-selected): {example}')


def load_source(name, args):
    import torch
    script = script_for(name, args.home)
    module_name = '_scad_profiling_' + script.stem
    spec = importlib.util.spec_from_file_location(module_name, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    inner = SOURCES[name][1]
    if inner not in mod.SPECS:
        raise RuntimeError('Source script %s lacks %s' % (script, inner))
    cfg = dict(mod.SPECS[inner])
    return mod, inner, cfg, default_sources(script.name, args.home)


def state_of(ckpt):
    import torch
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'model_state_dict'):
            if isinstance(ckpt.get(key), dict):
                return ckpt[key]
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError('No recognizable fine-tuned state_dict in checkpoint')


def validate_checkpoint_identity(name, ckpt, cfg):
    if not isinstance(ckpt, dict):
        return
    meta = ckpt.get('configuration', {})
    if not isinstance(meta, dict):
        return
    reported = meta.get('model')
    expected = SOURCES[name][1]
    if reported and reported != expected:
        raise ValueError('%s: checkpoint metadata names %s, expected %s' % (name, reported, expected))
    for key, wanted in (('frames', cfg['frames']), ('input_width', cfg['width']), ('input_height', cfg['height'])):
        if key in meta and meta[key] != wanted:
            raise ValueError('%s: checkpoint %s=%s, but expected %s' % (name, key, meta[key], wanted))


def memory_clean():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * p / 100.0
    a, b = int(x), min(int(x) + 1, len(values)-1)
    return values[a] * (b-x) + values[b] * (x-a)


def normalize_input(u8_tensor, cfg):
    """Same fixed SCAD inference normalization used by original runners."""
    import torch
    x = u8_tensor.to('cuda', non_blocking=True).float().div_(255.0)
    mean = torch.tensor(cfg['mean'], device=x.device).view(1,1,3,1,1)
    std = torch.tensor(cfg['std'], device=x.device).view(1,1,3,1,1)
    return ((x - mean) / std).permute(0, 2, 1, 3, 4).contiguous()


def model_output(model, gpu_input):
    out = model(gpu_input)
    if not hasattr(out, 'shape') or tuple(out.shape) != (1, 2):
        raise RuntimeError('Model did not return [1,2] binary logits: %s' % (getattr(out,'shape',type(out)),))
    if not out.isfinite().all().item():
        raise RuntimeError('Non-finite logits')
    return out


def measure_flops(model, x, amp_dtype):
    """Operator-counted forward FLOPs, plus fused SDPA QK/AV matmuls when visible.

    This is *not* an exhaustive total for every custom attention kernel; output
    explicitly records counting limitations, making it safe for exploratory work.
    """
    import torch
    import torch.nn.functional as F
    from torch.profiler import profile, ProfilerActivity
    extra = [0]
    sdpa_calls = [0]
    original = F.scaled_dot_product_attention
    def tracked_sdpa(q, k, v, *pos, **kw):
        if q.ndim >= 3 and k.ndim >= 3 and v.ndim >= 3:
            outer = 1
            for d in q.shape[:-2]: outer *= int(d)
            nq, nk, dq, dv = int(q.shape[-2]), int(k.shape[-2]), int(q.shape[-1]), int(v.shape[-1])
            extra[0] += 2 * outer * nq * nk * (dq + dv)
            sdpa_calls[0] += 1
        return original(q, k, v, *pos, **kw)
    try:
        F.scaled_dot_product_attention = tracked_sdpa
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False, with_flops=True) as prof:
            with torch.inference_mode(), torch.autocast('cuda', dtype=amp_dtype):
                model(x)
            torch.cuda.synchronize()
        flops = sum(int(evt.flops or 0) for evt in prof.key_averages())
        return {'operator_counted_gflops': round((flops + extra[0]) / 1e9, 4),
                'base_operator_gflops': round(flops / 1e9, 4),
                'sdpa_extra_gflops': round(extra[0] / 1e9, 4),
                'sdpa_calls': sdpa_calls[0],
                'method': 'torch.profiler with_flops + SDPA QK/AV estimate; incomplete for unsupported/custom/fused ops'}
    except Exception as ex:
        return {'operator_counted_gflops': None, 'flop_error': str(ex)}
    finally:
        F.scaled_dot_product_attention = original
        memory_clean()


def run_single(args):
    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; GPU needed for latency')
    torch.set_num_threads(min(2, os.cpu_count() or 2))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    name = args.single
    path = discover(name, args)
    source, inner, cfg, source_dir = load_source(name, args)
    print('BUILD', name, '| source', source.__file__, '| fine-tuned checkpoint', path, flush=True)
    # Instantiate using exactly the original model builder, then REPLACE all
    # parameters with our saved fine-tuned checkpoint. Load must be strict.
    torch.manual_seed(42)
    model, pretrained_desc = source.build_model(inner, source_dir)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_identity(name, checkpoint, cfg)
    state = state_of(checkpoint)
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    params = sum(p.numel() for p in model.parameters())
    if isinstance(checkpoint, dict):
        meta = checkpoint.get('configuration', {})
        if isinstance(meta, dict) and meta.get('parameters') and params != int(meta['parameters']):
            raise RuntimeError('Model size differs from saved checkpoint metadata')
    print('CHECKPOINT LOADED STRICTLY. Params=%.2fM' % (params/1e6), flush=True)
    files = source.scan(args.data, 'test')
    if len(files) != 232:
        raise RuntimeError('Expected 232 held-out test videos; found %d' % len(files))
    # Representative, predetermined spread over sorted filenames, never sampled
    # or selected by labels or performance.
    indices = np.linspace(0, len(files)-1, args.videos).round().astype(int).tolist()
    videos = [files[i][0] for i in indices]
    rgb = source.decode_one(videos[0], cfg['frames'], cfg['width'], cfg['height'])
    cpu_input = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0,3,1,2).unsqueeze(0).contiguous()
    with torch.inference_mode():
        gpu_input = normalize_input(cpu_input, cfg)
        with torch.autocast('cuda', dtype=amp_dtype):
            model_output(model, gpu_input)
    print('INPUT', cfg['frames'], 'x', cfg['width'], 'x', cfg['height'], '| precision', amp_dtype, flush=True)
    flops = measure_flops(model, gpu_input, amp_dtype)
    print('FLOPS', flops, flush=True)
    torch.cuda.reset_peak_memory_stats()
    model_times = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=amp_dtype):
        for _ in range(args.warmup):
            model(gpu_input)
        torch.cuda.synchronize()
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            start.record()
            model(gpu_input)
            finish.record()
            finish.synchronize()
            model_times.append(start.elapsed_time(finish))
    gpu_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print('INFERENCE model-only median=%.2f ms, p95=%.2f ms' %
          (statistics.median(model_times), percentile(model_times,95)), flush=True)
    # End-to-end times are wall clock (not CUDA event), including sequential CPU
    # decode, letterboxing, tensor conversion, H2D, normalization and inference.
    def e2e_one(video):
        image = source.decode_one(video, cfg['frames'], cfg['width'], cfg['height'])
        cpu = torch.from_numpy(np.ascontiguousarray(image)).permute(0,3,1,2).unsqueeze(0).contiguous()
        with torch.inference_mode(), torch.autocast('cuda', dtype=amp_dtype):
            logits, _ = source.forward_batch(model, cpu, torch.zeros(1,dtype=torch.long), cfg, False)
        torch.cuda.synchronize()
        if tuple(logits.shape) != (1,2):
            raise ValueError('Bad end-to-end output')
    for i in range(args.e2e_warmup):
        e2e_one(videos[i % len(videos)])
    e2e_times = []
    for i in range(args.e2e_repeats):
        start = time.perf_counter()
        e2e_one(videos[i % len(videos)])
        e2e_times.append((time.perf_counter()-start)*1000.0)
    record = dict(model=name, model_source=source.__file__, checkpoint=str(path),
                  params=params, parameters_m=round(params/1e6,4),
                  frames=cfg['frames'],width=cfg['width'],height=cfg['height'],
                  precision=str(amp_dtype),batch_size=1,
                  model_latency_median_ms=round(statistics.median(model_times),3),
                  model_latency_mean_ms=round(statistics.mean(model_times),3),
                  model_latency_p95_ms=round(percentile(model_times,95),3),
                  e2e_latency_median_ms=round(statistics.median(e2e_times),3),
                  e2e_latency_mean_ms=round(statistics.mean(e2e_times),3),
                  e2e_latency_p95_ms=round(percentile(e2e_times,95),3),
                  inference_gpu_peak_allocated_mib=round(gpu_peak,1),
                  device=torch.cuda.get_device_name(),
                  warmup=args.warmup,repeats=args.repeats,e2e_repeats=args.e2e_repeats,
                  e2e_videos=len(videos),pretrained_builder=str(pretrained_desc),
                  checkpoint_best_epoch=checkpoint.get('epoch') if isinstance(checkpoint,dict) else None,
                  **flops)
    out = args.results / (name + '_profile.json')
    out.write_text(json.dumps(record, indent=2, default=str))
    print('PROFILE RESULT', json.dumps(record, default=str), flush=True)
    del model, checkpoint, gpu_input, cpu_input
    memory_clean()
    return record


def audit(args):
    status = []
    for name in SOURCES:
        try:
            path = discover(name, args)
            status.append({'model':name, 'status':'ready', 'checkpoint':str(path), 'size_gib':round(path.stat().st_size/1024**3,2)})
        except Exception as exc:
            status.append({'model':name, 'status':'missing_or_ambiguous', 'error':str(exc)})
    for row in status:
        print('%-25s %-21s %s' % (row['model'],row['status'],row.get('checkpoint',row.get('error'))), flush=True)
    (args.results/'checkpoint_audit.json').write_text(json.dumps(status,indent=2))
    return all(x['status']=='ready' for x in status)


def save_csv(args):
    names = list(SOURCES)
    rows=[]
    for n in names:
        f=args.results/(n+'_profile.json')
        if f.exists():
            try:rows.append(json.loads(f.read_text()))
            except Exception:pass
    if rows:
        fields=['model','parameters_m','operator_counted_gflops','base_operator_gflops','sdpa_extra_gflops',
                'model_latency_median_ms','model_latency_p95_ms','e2e_latency_median_ms',
                'e2e_latency_p95_ms','inference_gpu_peak_allocated_mib','frames','width','height',
                'device','precision','checkpoint']
        with (args.results/'scad17_profile_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            writer.writeheader();writer.writerows(rows)
        print('WROTE',args.results/'scad17_profile_summary.csv', '| profiles:',len(rows),flush=True)


def benchmark(args):
    if not audit(args):
        raise RuntimeError('Checkpoint audit failed. Ensure one .pt file in each model directory before profiling.')
    failed = []
    for name in SOURCES:
        dest = args.results/(name+'_profile.json')
        if dest.exists() and not args.rerun_profile:
            print('SKIP already profiled:',name,flush=True)
            continue
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage','single',
             '--single',name,'--home',str(args.home),'--data',str(args.data),
             '--checkpoints',str(args.checkpoints),'--results',str(args.results),
             '--warmup',str(args.warmup),'--repeats',str(args.repeats),
             '--e2e-warmup',str(args.e2e_warmup),'--e2e-repeats',str(args.e2e_repeats),
             '--videos',str(args.videos)]
        print('\nPROFILE', name, flush=True)
        with (args.results/(name+'_profile.log')).open('w') as f:
            proc=subprocess.Popen(cmd,cwd=args.home,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            code=proc.wait()
        if code:
            failed.append(name)
            print('FAILED:',name,'see',args.results/(name+'_profile.log'),flush=True)
        save_csv(args)
    if failed:
        (args.results/'profile_failures.json').write_text(json.dumps(failed,indent=2))
        raise RuntimeError('Failed profiles: '+', '.join(failed))
    print('COMPLETE: SCAD profiling summary at',args.results/'scad17_profile_summary.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description='Train missing SCAD checkpoints; profile all 17 variants on L40S')
    p.add_argument('--stage',choices=['train','audit','inventory','profile','all','single'],default='all')
    p.add_argument('--single',choices=list(SOURCES))
    p.add_argument('--home',type=Path,default=DEFAULT_HOME)
    p.add_argument('--data',type=Path)
    p.add_argument('--checkpoints',type=Path)
    p.add_argument('--results',type=Path)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--warmup',type=int,default=30)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--e2e-warmup',type=int,default=3)
    p.add_argument('--e2e-repeats',type=int,default=40)
    p.add_argument('--videos',type=int,default=8)
    p.add_argument('--rerun-profile',action='store_true')
    args=p.parse_args()
    args.home=args.home.expanduser().resolve()
    args.data=(args.data or args.home/'accident').expanduser().resolve()
    args.checkpoints=(args.checkpoints or args.home/'scad'/'models').expanduser().resolve()
    args.results=(args.results or args.home/'scad'/'profiling_17').expanduser().resolve()
    args.results.mkdir(parents=True,exist_ok=True)
    if args.videos<1 or args.videos>232: p.error('--videos must be 1..232')
    if args.stage=='single' and not args.single: p.error('--single required for stage single')
    if args.stage in ('train','all'):
        if not all((args.data/x).is_dir() for x in ('train','val','test')):
            raise RuntimeError('Cannot find train/val/test below '+str(args.data))
        train_missing(args)
    if args.stage=='inventory':
        paths=checkpoint_inventory(str(args.home),str(args.checkpoints))
        print('FOUND',len(paths),'checkpoint .pt files:',flush=True)
        for f in paths:print(' ',f,flush=True)
    if args.stage=='audit':
        if not audit(args): sys.exit(2)
    if args.stage in ('profile','all'):
        benchmark(args)
    if args.stage=='single':
        run_single(args)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('PIPELINE ERROR:',exc, file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.exit(2)
