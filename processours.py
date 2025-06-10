import argparse

p = argparse.ArgumentParser()
p.add_argument('file', nargs="+", help="STDOUT from our CriticalPoints run")
args = p.parse_args()

suffix100s = ['ALL Critical Points',]

averaging = dict()

for fname in args.file:
    print(fname)
    with open(fname, 'r') as f:
        lines = f.readlines()
        keep = list()
        for line in lines:
            if any([_ in line for _ in suffix100s]):
                keep.append(line.rstrip())

    total_time = 0.0
    for line in keep:
        for maybe in suffix100s:
            if maybe in line:
                what = maybe
                section = line[line.rindex(' ')+1:]
                break
        print("\t", what, '---', section)
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

