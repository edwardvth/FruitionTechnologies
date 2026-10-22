"""Safe extracted-function probes; never loads checkpoints or imports project modules."""
import ast
import json
import subprocess
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
def extract(path, name, namespace):
    source = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT, text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]
class Writer:
    def add_scalar(self, *args): pass
class Model:
    def eval(self): pass
    def __call__(self, images, targets):
        # False positive nowhere near the annotated box.
        return {'detections': torch.tensor([[[300.,300.,400.,400.,.99,1.]]])}
validate = extract('goose_detection/scripts/train.py', 'validate', {'torch':torch})
target = {'bbox':[torch.tensor([[1.,1.,10.,10.]])], 'cls':[torch.tensor([1])],
          'img_size':torch.tensor([[512.,512.]]), 'img_scale':torch.tensor([1.])}
score = validate(Model(), [(torch.zeros(1,3,512,512), target)], 'cpu', 1, Writer())
run = extract('goose_detection/scripts/test_model.py', 'run_inference', {'torch':torch, 'IMG_SIZE':512})
result = run(lambda *args, **kwargs: torch.tensor([[[10.,20.,110.,220.,.9,1.]]]),
             torch.zeros(1,3,512,512), torch.device('cpu'), 2., 1., .5)
out = {'false_positive_only_validation_score':score,
       'zero_iou_expected':True,
       'upstream_xyxy_input':[10,20,110,220], 'scale_x':2, 'scale_y':1,
       'expected_output_box':[20,20,220,220], 'actual_output_box':list(result[0][:4])}
assert score > .98
assert list(result[0][:4]) == [40,10,440,110]
(ROOT/'reports/audit-2026-10-22/model-evidence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
