from collections import defaultdict
import argparse

p = argparse.ArgumentParser()
p.add_argument('file', nargs="+", help="STDOUT from ACTOPO TTK ScalarFieldCriticalPoints run")
args = p.parse_args()

search_strs = ['Timer[Preprocessing]',
               'Timer[Preprocessing Cells]',
               'Timer[ACTOPO alg]']
exclude = defaultdict(list)

averaging = dict()
breakdown = dict()
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
    if 'iter' in fname:
        aname = fname[:fname.index('iter')]
        if aname not in averaging:
            averaging[aname] = list()
            breakdown[aname] = dict((w, [0.0]) for w in search_strs)
        else:
            for w in breakdown[aname]:
                breakdown[aname][w].append(0.0)

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
            breakdown[aname][what][-1] += breakdown_time
            if what == 'Timer[Preprocessing Cells]':
                breakdown[aname]['Timer[ACTOPO alg]'][-1] -= breakdown_time
            else:
                total_time += breakdown_time
        except ValueError:
            print("\t", line)

    print(f"Total time for {fname}: {total_time}")
    if 'iter' in fname:
        averaging[aname].append(total_time)

for aname, times in averaging.items():
    print(f"Average time for {aname}: {sum(times)/len(times):.3f}")
    for what in breakdown[aname].keys():
        print('\t'+f"{what}: {sum(breakdown[aname][what])/len(breakdown[aname][what]):.3f}")

