"""Final training entry point: initial training, continuation, and validation selection.

Use --stage initial (default), --stage continue, or --stage select.
Each stage accepts --help. All model selection uses validation only.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from student import build_model

def train_initial():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = .001 * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)



def train_continuation():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=23)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cpu')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--lr',type=float,default=.0006)
    p.add_argument('--weight-decay',type=float,default=.1)
    p.add_argument('--eval-every',type=int,default=600)
    p.add_argument('--average-start',type=int,default=3000)
    p.add_argument('--average-every',type=int,default=100)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.eval_every,a.average_every)<1:
        p.error('Counts must be positive.')
    if not a.resume and a.run_dir.exists() and any(a.run_dir.iterdir()):
        p.error('Use a new run directory or explicitly --resume.')
    device,precision=setup(a.device,a.precision,a.threads)
    torch.manual_seed(a.seed)
    parent=torch.load(a.parent,map_location='cpu',weights_only=True)
    assert parent['protocol']==PROTOCOL
    model,source_hash=make_model(parent['implementation'],parent['config'],device)
    model.load_state_dict(parent['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    rng=torch.Generator().manual_seed(a.seed)
    data=load_data(); tokens=data['train'][0].to(device)
    average=None; average_count=0; start_step=0; best=float('inf')
    history=[]; train_seconds=0.; validation_seconds=0.
    a.run_dir.mkdir(parents=True,exist_ok=True)
    parent_hash=sha(a.parent)
    lineage={'path':str(a.parent),'sha256':parent_hash,'processed_targets':parent['train_tokens']}
    recipe={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'}
    if a.resume:
        last=torch.load(a.run_dir/'resume.pt',map_location='cpu',weights_only=True)
        if last['recipe']!=recipe or last['parent_sha256']!=parent_hash:
            raise ValueError('Resume must use identical recipe and parent.')
        model.load_state_dict(last['model']); optimizer.load_state_dict(last['optimizer'])
        rng.set_state(last['rng']); torch.set_rng_state(last['torch_rng'])
        start_step=last['step']; best=last['best']; history=last['history']
        average={k:v.to(device) for k,v in last['average'].items()} if last['average'] is not None else None; average_count=last['average_count']
        train_seconds=last['train_seconds']; validation_seconds=last['validation_seconds']
    def save_checkpoint(path,step,variant):
        torch.save({'protocol':PROTOCOL,'implementation':parent['implementation'],'config':parent['config'],
                    'model':model.state_dict(),'seed':a.seed,'train_tokens':parent['train_tokens']+step*a.batch_size*256,
                    'parent_checkpoints':[lineage],'extension_steps':step,'variant':variant,'recipe':recipe},path)
    def event(row):
        with (a.run_dir/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    for step in range(start_step,a.steps):
        began=time.perf_counter()
        # Exactly the official training-only random-window sampling support.
        starts=torch.randint(len(tokens)-257,(a.batch_size,),generator=rng).to(device)
        batch=tokens[starts[:,None]+torch.arange(257,device=device)]
        lr=a.lr*min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        for group in optimizer.param_groups: group['lr']=lr
        model.train(); optimizer.zero_grad(set_to_none=True)
        with autocast(device,precision):
            loss=F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        done=step+1
        if done>=a.average_start and done%a.average_every==0:
            with torch.no_grad():
                if average is None:
                    average={k:v.detach().clone() for k,v in model.state_dict().items()}
                else:
                    for k,v in model.state_dict().items():
                        average[k].lerp_(v,1./(average_count+1))
                average_count+=1
        if device.type=='cuda': torch.cuda.synchronize(device)
        train_seconds+=time.perf_counter()-began
        if done%100==0 or done==a.steps:
            event({'step':done,'loss':loss.item(),'learning_rate':lr,'train_seconds':train_seconds,'processed_targets':parent['train_tokens']+done*a.batch_size*256})
        if done%a.eval_every==0 or done==a.steps:
            variants=['live']+(['average'] if average is not None else [])
            live={k:v.detach().clone() for k,v in model.state_dict().items()}
            for variant in variants:
                if variant=='average': model.load_state_dict(average)
                validation=score(model,*data['validation'],device,'fp32')
                validation.pop('window_nll_nats'); validation_seconds+=validation['seconds']
                row={'step':done,'variant':variant,**validation}
                history.append(row); event({'validation':row})
                if validation['bpb']<best:
                    best=validation['bpb']; save_checkpoint(a.run_dir/'checkpoint.pt',done,variant)
            model.load_state_dict(live)
            save_checkpoint(a.run_dir/'last_checkpoint.pt',done,'live')
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.get_state(),'torch_rng':torch.get_rng_state(),
                   'step':done,'best':best,'history':history,'average':average,'average_count':average_count,
                   'train_seconds':train_seconds,'validation_seconds':validation_seconds,'recipe':recipe,'parent_sha256':parent_hash}
            torch.save(state,a.run_dir/'resume.pt')
            metrics={'protocol':PROTOCOL,'recipe':recipe,'parent_checkpoints':[lineage],'total_extension_steps':done,
                     'total_search_targets':parent['train_tokens']+done*a.batch_size*256,'train_seconds':train_seconds,
                     'validation_seconds':validation_seconds,'best_validation_bpb':best,'history':history,
                     'parameters':sum(v.numel() for v in model.parameters()),'trainer_sha256':sha(Path(__file__)),
                     'implementation_sha256':source_hash,'checkpoint_sha256':sha(a.run_dir/'checkpoint.pt'),'torch_version':str(torch.__version__),'precision':precision,**device_metrics(device)}
            (a.run_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    event({'completed':True,'best_validation_bpb':best,'train_seconds':train_seconds})


def select_copy():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): p.error('Use a fresh output directory.')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    device,_=setup(a.device,'fp32',4)
    parent=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    data=load_data()
    rows=[]; best=float('inf'); best_config=None
    grid=[(0.,2,2.)]+[(alpha,order,2.) for order in (1,2,3) for alpha in (.1,.25,.4)]
    for alpha,order,smoothing in grid:
        config=parent['config']|dict(copy_alpha=alpha,copy_order=order,copy_smoothing=smoothing)
        model=build_model(config).to(device); model.load_state_dict(parent['model'])
        result=score(model,*data['validation'],device,'fp32'); result.pop('window_nll_nats')
        row={'config':config,**result}; rows.append(row)
        if result['bpb']<best: best=result['bpb']; best_config=config
        (a.output_dir/'validation_grid.json').write_text(json.dumps({'selection_split':'validation','parent_checkpoint':str(a.checkpoint),'parent_sha256':sha(a.checkpoint),'rows':rows,'best_bpb':best,'best_config':best_config},indent=2)+'\n')
        print(json.dumps(row),flush=True)
        del model
    checkpoint=dict(parent)
    checkpoint.update(implementation='student',config=best_config,selection_split='validation',selection_parent_sha256=sha(a.checkpoint),additional_training_targets=0)
    torch.save(checkpoint,a.output_dir/'checkpoint.pt')
    print(json.dumps({'selected_config':best_config,'validation_bpb':best,'checkpoint_sha256':sha(a.output_dir/'checkpoint.pt')}),flush=True)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stage', choices=['initial', 'continue', 'select'], default='initial')
    stage, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    {'initial': train_initial, 'continue': train_continuation, 'select': select_copy}[stage.stage]()

if __name__ == '__main__':
    main()
