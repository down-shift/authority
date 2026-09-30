#!/usr/bin/env python3
"""Run the predefined small engineering pilot (4 worlds/family) or config pilot."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_indirection_experiment import main
if __name__=='__main__': main()
