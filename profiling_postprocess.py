import pandas as pd

import argparse
import pathlib
from io import StringIO

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('infile', type=pathlib.Path, help="File to read as input")
    prs.add_argument('outfile', type=pathlib.Path, help="File to export to")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

def main():
    args = parse()
    with open(args.infile,'r') as f:
        lines = f.readlines()
    # Parsing
    mode = None
    records = list()
    general_records = list()
    buffer = list()
    for line in lines:
        if 'Approximated max' in line:
            adjacency = int(line.rstrip().rsplit(' ',1)[1])
            continue
        # Start general profiling
        if ('Type' in line) and ('Time(%)' in line) and ('Time' in line) and \
           ('Calls' in line) and ('Avg' in line) and ('Min' in line) and \
           ('Max' in line) and ('Name' in line):
            mode = "GeneralProfiling"
            general_records = list()
            continue
        # Start metric profiling
        if line.rstrip() == '"Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"':
            mode = "MetricProfiling"
            buffer = [line]
            continue
        # Handle general profiling
        if mode == "GeneralProfiling":
            # Exit general profiling
            if line.rstrip() == 'END_GENERAL_PROFILING':
                mode = None
                records.append(pd.DataFrame(general_records))
                continue
            # Parse
            if ':' in line:
                _type, metrics_and_name = line.lstrip().rstrip().split(':',1)
            else:
                metrics_and_name = line.rstrip().lstrip()
            try:
                metrics, name = metrics_and_name[:-1].split('[',1)
            except:
                if '(' in metrics_and_name:
                    metrics_func,sig = metrics_and_name.split('(',1)
                    metrics, name = metrics_func.rsplit(' ',1)
                    name += '('+sig
                else:
                    metrics, name = metrics_and_name.rsplit(' ',1)
            pct_time,time,calls,avg,_min,_max = [_ for _ in metrics.split(' ') if _ != '']
            record = {'type': _type,
                      'name': name,
                      'time(%)': pct_time,
                      'time(s)': time,
                      'calls': calls,
                      'avg': avg,
                      'min': _min,
                      'max': _max,
                      }
            general_records.append(record)
        # Handle metric profiling
        if mode == "MetricProfiling":
            if line.rstrip() == "END_METRIC_PROFILING":
                mode = None
                record = pd.read_csv(StringIO("\n".join(buffer)))
                records.append(record)
                continue
            buffer.append(line)
    first_time = True
    for record in records:
        record.to_csv(args.outfile, index=False, mode='a' if not first_time else 'w')
        first_time = False
        with open(args.outfile,'a') as f:
            f.write('\n')

if __name__ == '__main__':
    main()

