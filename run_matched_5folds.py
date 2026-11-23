#!/usr/bin/env python3
"""Sequential A/B runs on phase7b fixed-test folds; dry-run unless --run."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from clusterprep.matched_runner import load_plan, read, device_name, training_configuration, training_budget

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'clusterprep/configs/phase7b_multimodal'
REQUIRED=('best.pt','history.csv','validation_predictions.csv',
          'validation_metrics.json','thresholds.json','metrics.json')

def completed(path,plan,training,budget=None):
    try:
        c=read(path/'request.json'); marker=read(path/'completed.json')
        result=Path(marker['run_dir'])
        if result.parent.resolve()!=path.resolve():return False
        status=read(result/'status.json')
        return (marker['pipeline']=='historical_matched_v1'
                and status['status']=='completed' and status.get('test_evaluated') is False
                and c['plan']==plan and c['training']==training
                and (budget is None or c.get('budget')==budget)
                and all((result/f).is_file() and (result/f).stat().st_size for f in REQUIRED))
    except (OSError,ValueError,KeyError):return False

def jobs(source,seed):
    plans=[load_plan(e,f,seed,source) for e in 'AB' for f in range(5)]
    external=set(plans[0]['partition']['test'])
    for plan in plans:
        p=plan['partition']
        if read(source/f"fold_{plan['fold']}.json")!=p:raise ValueError('Individual fold differs from runner manifest')
        if set(p['test'])!=external or external & set(p['train']+p['validation']):raise ValueError('External test leakage')
        # CHANDRA is optional in multimodal mode; load_plan validates the full population.
    return plans

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true',help='Actually launch training; otherwise verify and display only')
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-epochs',type=int,default=200)
    p.add_argument('--epoch-mode',choices=['full'],default='full')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/phase7b_multimodal')
    args=p.parse_args(argv)
    device=device_name(args.device)
    plans=jobs(SOURCE,args.seed)  # validate all ten before starting any
    for index,plan in enumerate(plans,1):
        e=plan['experiment'];fold=plan['fold']
        training=asdict(training_configuration(plan,device,args.max_epochs))
        budget=training_budget(plan,args.max_epochs,args.epoch_mode)
        parent=args.output.resolve()/f'{e}_fold{fold}_seed{args.seed}'
        attempts=sorted(parent.glob('attempt_*'))
        if any(completed(a,plan,training,budget) for a in attempts):
            print(f'[{index}/10] {e} fold {fold}: COMPLETE — skipped',flush=True);continue
        number=1
        while (parent/f'attempt_{number:03d}').exists() or (parent/f'attempt_{number:03d}.log').exists():number+=1
        target=parent/f'attempt_{number:03d}'
        command=[sys.executable,'-u','-m','clusterprep.matched_runner','train','--experiment',e,
                 '--fold',str(fold),'--seed',str(args.seed),'--device',device,
                 '--max-epochs',str(args.max_epochs),'--source',str(SOURCE),'--epoch-mode',args.epoch_mode,'--output',str(target)]
        print(f'[{index}/10] {e} ({plan["product"]}) fold {fold}: '+('START' if args.run else 'READY (dry-run)'),flush=True)
        print(f"  {args.epoch_mode}: {budget['train_examples']} images, {budget['steps_per_epoch']} batches/epoch",flush=True)
        if not args.run:continue
        parent.mkdir(parents=True,exist_ok=True)
        log=target.with_suffix('.log')
        print(f'  Log: {log}',flush=True)
        try:
            with log.open('w') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode or not completed(target,plan,training,budget):
                print(f'ERROR: {e} fold {fold}; stopped. See {log}',file=sys.stderr,flush=True)
                return result.returncode if result.returncode>0 else 1
        except KeyboardInterrupt:
            print(f'INTERRUPTED: {e} fold {fold}; completed runs retained.',file=sys.stderr)
            return 130
        print(f'[{index}/10] {e} fold {fold}: COMPLETE',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
