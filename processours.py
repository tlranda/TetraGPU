import argparse

p = argparse.ArgumentParser()
p.add_argument('file', nargs="+", help="STDOUT from our CriticalPoints run")
args = p.parse_args()

# 'ALL Critical Points', 
suffix100s = ["SFCP kernel"]
exclude = {'SFCP kernel': ["Setup for"],}

averaging = dict()

for fname in args.file:
    print(fname)
    with open(fname, 'r') as f:
        lines = f.readlines()
        keep = list()
        for line in lines:
            for maybe in suffix100s:
                if maybe not in line:
                    continue
                excluded = False
                for exclusion in exclude[maybe]:
                    if exclusion in line:
                        excluded = True
                        break
                if not excluded:
                    keep.append(line.rstrip())

    total_time = 0.0
    for line in keep:
        for maybe in suffix100s:
            if maybe in line:
                what = maybe
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

