"""Safe source-extracted probes; no project imports, model loads, or real GPIO/camera.
Run from repository root: python3 reports/audit-2026-10-22/runtime_probes.py
Reads audited source from Git HEAD, so results correspond to that revision.
"""
import ast
import json
import subprocess
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def source(path):
    return subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT, text=True)

def extract(path, names, namespace):
    tree = ast.parse(source(path))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, 'exec'), namespace)

class EndProbe(Exception):
    pass

class Scores(list):
    def max(self):
        return max(self)

class Clock:
    def perf_counter(self): return 100.0
    def time(self): return 100.0

results = []
for scenario in ('dropped_frames', 'next_inference', 'person_detected'):
    events = []
    ns = {'CLIP_FPS': 15, 'ALERT_COOLDOWN_SEC': 1, 'time': Clock(),
          'GPIO_PIN': 18, '_laser_state': False, '_last_sim_print': 0}
    class GPIO:
        HIGH = 1
        LOW = 0
        def output(self, pin, value): events.append(value)
    ns['_gpio_handle'] = GPIO()
    observations = []
    class Capture:
        reads = 0
        def isOpened(self): return True
        def read(self):
            self.reads += 1
            if scenario == 'dropped_frames' and self.reads in (2, 3):
                observations.append(ns['_laser_state'])
                return False, None
            if self.reads > (3 if scenario == 'dropped_frames' else 2):
                observations.append(ns['_laser_state'])
                raise EndProbe()
            return True, object()
        def release(self): pass
    cap = Capture()
    ns['cv2'] = types.SimpleNamespace(VideoCapture=lambda *args: cap, CAP_GSTREAMER=1,
                                     FONT_HERSHEY_SIMPLEX=0, putText=lambda *args: None)
    ns['gstreamer_pipeline'] = lambda: 'mock'
    ns['setup_gpio'] = lambda: None
    ns['draw_geese'] = lambda frame, *args: frame
    ns['draw_people'] = lambda frame, *args: frame
    ns['draw_safety_banner'] = lambda frame: frame
    class Tracker:
        count = 0
        def track(self, frame):
            self.count += 1
            if scenario == 'next_inference' and self.count == 2:
                observations.append(ns['_laser_state'])
                raise EndProbe()
            return [[0, 0, 1, 1]], Scores([0.9]), [1]
    class Safety:
        count = 0
        def detect(self, frame):
            self.count += 1
            return [[0, 0, 1, 1]] if scenario == 'person_detected' and self.count == 2 else []
    extract('goose_detection/inference/inference_pipeline.py',
            {'run_camera_loop', '_set_laser', 'laser_on', 'laser_off'}, ns)
    try:
        ns['run_camera_loop'](Tracker(), Safety(), save_clips=False, headless=True)
    except EndProbe:
        pass
    expected = [True, True, True] if scenario == 'dropped_frames' else [True] if scenario == 'next_inference' else [False]
    assert observations == expected, (scenario, observations)
    assert events == [1, 0], events
    results.append({'probe': scenario, 'laser_state_observed_before_probe_exit': observations,
                    'mock_GPIO_output_sequence_including_finally': events, 'assertions_passed': True})

ns = {}
extract('goose_detection/inference/clip_recorder.py', {'ClipRecorder'}, ns)
Recorder = ns['ClipRecorder']
r = object.__new__(Recorder)
r._stop_sentinel = object()
class MockQueue:
    def get(self): return (None, False, 0, 0, [], 1)
    def put(self, *args, **kwargs):
        results.append({'probe': 'close_queue_put', 'explicit_timeout': kwargs.get('timeout'),
                        'blocking_default': kwargs.get('block', True)})
        raise EndProbe()
r._queue = MockQueue()
def failing_process(*args): raise OSError('mock encoder failure')
r._process = failing_process
try:
    r._worker_loop()
except OSError:
    results.append({'probe': 'worker_encoder_exception', 'exception_escapes_worker': True})
else:
    raise AssertionError('Expected unhandled encoder error')
try: r.close()
except EndProbe: pass
print(json.dumps({'head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                  'results': results}, indent=2))
