"""Run run_vmc12_n06 through the shared archive launcher."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run import main

if __name__=="__main__":
    main(["run_vmc12_n06"]+sys.argv[1:])
