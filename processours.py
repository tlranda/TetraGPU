from collections import defaultdict
import argparse

p = argparse.ArgumentParser()
p.add_argument('file', nargs="+", help="STDOUT from our CriticalPoints run")
args = p.parse_args()

"""
⏳ Timer[VTK Preprocessing] Elapsed time for interval 0(0, 1): 0.000461
⏳ Timer[ALL Critical Points] Elapsed time for interval 0(0, 1): 0.070294
"""
# Substrings to match for lines that end in floating-point value to add up
search_strs = ["Timer[VTK Preprocessing]",
               "Timer[ALL Critical Points]"]
# Key (entry of search_strs) | Value (substring that lets you know to exclude it)
exclude = defaultdict(list)

averaging = dict()
for fname in args.file:
    print(fname)
    with open(fname, 'r') as f:
        lines = f.readlines()
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

    total_time = 0.0
    for line in keep:
        for trigger in search_strs:
            if trigger in line:
                what = trigger
                section = line[line.rindex(' ')+1:]
                break
        print("\t"+f"{what} ({line}) --- {section}")
        try:
            total_time += float(section)
        except ValueError:
            print("\t", line)

    print(f"Total time for {fname}: {total_time}")
    if 'iter' in fname:
        aname = fname[:fname.index('iter')]
        if aname not in averaging:
            averaging[aname] = list()
        averaging[aname].append(total_time)

for aname, times in averaging.items():
    print(f"Average time for {aname}: {sum(times)/len(times)}")

