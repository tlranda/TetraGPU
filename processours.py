from collections import defaultdict
import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument('file', nargs="*", default=None, help="STDOUT from our CriticalPoints run (or stdin if none specified)")
p.add_argument('--represents', choices=['all','main','worker'], default='all', help='What is represented by the input files (all output, main()\'s output, parallel_work() output of thread)')
args = p.parse_args()
if len(args.file) == 0:
    args.file.append(None)

"""
⏳ Timer[VTK Preprocessing] Elapsed time for interval 0(0, 1): 0.000461
⏳ Timer[ALL Critical Points] Elapsed time for interval 0(0, 1): 0.070294
"""
# Substrings to match for lines that end in floating-point value to add up
if args.represents == 'all':
    search_strs = ["Timer[VTK Preprocessing]",
                   "Timer[ALL Critical Points]"]
elif args.represents == 'main':
    search_strs = ["Timer[ALL Critical Points]"]
elif args.represents == 'worker':
    search_strs = ["Timer[Parallel worker"]

# Key (entry of search_strs) | Value (substring that lets you know to exclude it)
exclude = defaultdict(list)

averaging = dict()
breakdown = dict()
for fname in args.file:
    if fname is None:
        lines = sys.stdin.readlines()
        fname = 'stdin'
    else:
        with open(fname, 'r') as f:
            lines = f.readlines()
    print(fname)
    keep = list()
    for line in lines:
        for trigger in search_strs:
            if trigger not in line:
                continue
            excluded = False
            for exclusion in exclude[trigger]:
                if exclusion in line:
                    excluded = True
                    break
            if not excluded:
                keep.append(line.rstrip())
    # Special fixup for worker
    if args.represents == 'worker':
        keep = keep[:-2]

    if 'iter' in fname:
        aname = fname[:fname.index('iter')]
    else:
        aname = fname
    if aname not in averaging:
        averaging[aname] = list()
        breakdown[aname] = dict()

    total_time = 0.0
    for line in keep:
        for trigger in search_strs:
            if trigger in line:
                what = trigger
                section = line[line.rindex(' ')+1:]
                break
        print("\t"+f"{what} ({line}) --- {section}")
        try:
            breakdown_time = float(section)
            if what not in breakdown[aname]:
                breakdown[aname][what] = list()
            breakdown[aname][what].append(breakdown_time)
            total_time += breakdown_time
        except ValueError:
            print("\t", line)

    print(f"Total time for {fname}: {total_time}")
    averaging[aname].append(total_time)

for aname, times in averaging.items():
    print(f"Average time for {aname}: {sum(times)/len(times):.3f}")
    for what in breakdown[aname].keys():
        print("\t"+f"{what}: {sum(breakdown[aname][what])/len(breakdown[aname][what]):.3f}")

