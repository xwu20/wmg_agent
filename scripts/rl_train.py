import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from utils.config_handler import cf
from utils.worker import Worker

parser = argparse.ArgumentParser('default arg parser')
parser.add_argument('--load_model', action='store', help='Input file for the full agent model.')
parser.add_argument('--save_model', action='store', help='Output file for the full agent model.')
parser.add_argument('--load_repres', action='store', help='Input file for the representation module.')
parser.add_argument('--save_repres', action='store', help='Output file for the representation module.')
parser.add_argument('--xtlib', action='store_true', default=False, help='Whether to use xtlib.')
args = parser.parse_args()

if args.xtlib:
    cf.consult_azure_for_config()

worker = Worker(args.load_model, args.save_model, args.load_repres, args.save_repres)
worker.train()
