import argparse

p = argparse.ArgumentParser()
p.add_argument('file', help="STDOUT from Tetra with critPoints MAIN run")
args = p.parse_args()

suffix100s = ['[1;33mVV[0m [GPU]:',
              'Allocate [1;36mCritical Points[0m memory:',
              'Run [1;36mCritical Points[0m algorithm:',
              ]

with open(args.file, 'r') as f:
    lines = f.readlines()
    keep = list()
    start = False
    for line in lines:
        if 'Memory usage' in line:
            keep.append(line.rstrip())
        elif any([_ in line for _ in suffix100s]):
            keep.append(line.rstrip())

total_time = 0.0
for line in keep:
    if 'Memory usage' in line:
        continue
    for maybe in suffix100s:
        if maybe in line:
            what = maybe
            section = line[line.rindex(' ')+1:]
            break
    print(what, '---', section)
    try:
        total_time += float(section)
    except ValueError:
        print(line)

print(f"Total time: {total_time}")

