"""Aggregate Table A5 metrics from three completed adaptation streams."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(folder):
    records = [json.loads((folder / f'run{seed}.json').read_text())
               for seed in range(3)]
    curves = []
    for seed, record in enumerate(records):
        checks = record['checkpoints']
        if (record['run'] != seed or record['arm'] != folder.name
                or record['shift'] != folder.parent.name
                or record['batch'] != 8 or record['eval_size'] != 100
                or record['eval_seed_start'] != 1000000):
            raise ValueError(f"Inconsistent protocol in {folder}/run{seed}.json")
        if [c['episode'] for c in checks] != list(range(0, 501, 50)):
            raise ValueError(f"Incomplete curve in {folder}/run{seed}.json")
        if any(c['updates'] != 4*c['episode'] for c in checks):
            raise ValueError(f"Unexpected update budget in {folder}/run{seed}.json")
        curves.append([c['mean_end'] for c in checks])
    peak = max(range(1,11), key=lambda i:statistics.mean(c[i] for c in curves))
    best = [c[peak] for c in curves]
    auc = [sum((x+y)*25/500 for x,y in zip(c,c[1:])) for c in curves]
    return {'best_episode': peak*50, 'best_mean': statistics.mean(best),
            'best_sample_sd': statistics.stdev(best),
            'auc_mean': statistics.mean(auc), 'auc_sample_sd': statistics.stdev(auc),
            'best_per_run': best, 'auc_per_run': auc}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--shift', choices=['grid','wave'], required=True)
    args = p.parse_args()
    root = args.directory / args.shift
    result = {arm: summarize(root/arm) for arm in ('persistent','reset')}
    path = root/'comparison.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(path)
    print(json.dumps(result, indent=2))

if __name__=='__main__':
    main()
