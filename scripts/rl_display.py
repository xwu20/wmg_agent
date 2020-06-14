import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from utils.worker import Worker

parser = argparse.ArgumentParser('default arg parser')
parser.add_argument('--load_model', action='store', help='Input file for the full agent model.')
args = parser.parse_args()

worker = Worker(args.load_model, None, None, None)
worker.display()
