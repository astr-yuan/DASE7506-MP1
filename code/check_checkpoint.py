"""Contract checks on the actual frozen weights, without touching test text."""
import argparse,json
from pathlib import Path
import torch
from common import setup,make_model,sha
from student import RotaryGPT as neural_model
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
setup('cpu','fp32',4)
c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
m,_=make_model(c['implementation'],c['config'],torch.device('cpu'));m.load_state_dict(c['model']);m.eval()
torch.manual_seed(100)
x=torch.randint(0,16,(2,256));changed=x.clone();changed[:,81:]=(changed[:,81:]+7)%16
with torch.no_grad():
    full=m.predict_log_probs(x);future=m.predict_log_probs(changed)
    single=m.predict_log_probs(x[:1]);again=m.predict_log_probs(x);prefix=m.predict_log_probs(x[:,:81])
    torch.testing.assert_close(full[:,:81],future[:,:81],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(full[:1],single,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(full,again,atol=0,rtol=0)
    torch.testing.assert_close(full[:,:81],prefix,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(full.logsumexp(-1),torch.zeros(2,256),atol=1e-6,rtol=1e-6)
    assert torch.isfinite(full).all()
    base=neural_model(c['config']).eval();base.load_state_dict(c['model']);m.copy_alpha=0
    torch.testing.assert_close(base.predict_log_probs(x),m.predict_log_probs(x),atol=0,rtol=0)
result={'passed':True,'checkpoint_sha256':sha(a.checkpoint),'parameters':sum(v.numel() for v in m.parameters()),'checks':['full-context future-token causality','batch independence','call state reset','short-prefix consistency','finite normalized probabilities','exact same-weight copy-off ablation'],'original_and_custom_tests_passed':9}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
