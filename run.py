"""Unified launcher. python run.py --list; python run.py TASK [arguments]."""
from pathlib import Path
import os,sys,json,runpy,unittest,importlib

ROOT=Path(__file__).resolve().parent
SOURCE_DIRS=[ROOT/'sampling',ROOT/'calculations',ROOT/'calculations/workflows',ROOT/'tests']

def configure():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ.setdefault(key,'1')
    for path in [ROOT]+SOURCE_DIRS:
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    # Prefer the existing workspace runtime when present; portable installations use pip.
    bundled=ROOT.parent.parent/'tmp/plot-libs'
    if bundled.exists():sys.path.insert(0,str(bundled))
    os.chdir(ROOT)

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    tasks=json.loads((ROOT/'tasks.json').read_text(encoding='utf-8'))
    if not argv or argv[0] in ('--help','-h','--list'):
        print('Usage: python run.py TASK [arguments]')
        print('Special tasks: test, check-imports')
        for name,path in tasks.items():print(f'  {name}: {path}')
        return
    task=argv.pop(0)
    if task not in tasks and task not in ('test','check-imports'):
        raise SystemExit(f'Unknown task: {task}; use --list')
    configure()
    if task=='test':
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():raise SystemExit(1)
    elif task=='check-imports':
        for folder in SOURCE_DIRS:
            for path in sorted(folder.glob('*.py')):
                importlib.import_module(path.stem)
        print('All sampling, calculation, workflow and test imports passed.')
    else:
        target=ROOT/tasks[task]
        sys.argv=[str(target)]+argv
        runpy.run_path(str(target),run_name='__main__')

if __name__=='__main__':main()
