import argparse

p = argparse.ArgumentParser()
p.add_argument('file', nargs="+", help="STDOUT from ACTOPO TTK ScalarFieldCriticalPoints run")
args = p.parse_args()

suffix100s = ['Triangles and boundary triangles preconditioned in',
              'Time usage for computation:',
              ]

for fname in args.file:
    print(fname)
    with open(fname, 'r') as f:
        lines = f.readlines()
        keep = list()
        start = False
        for line in lines:
            if not start:
                if 'ScalarFieldCriticalPoints' not in line:
                    continue
                start = True
                continue
            elif 's|' in line:
                keep.append(line.rstrip())
            elif 'Memory usage' in line:
                keep.append(line.rstrip())
            elif any([_ in line for _ in suffix100s]):
                keep.append(line.rstrip())

    total_time = 0.0
    for line in keep:
        if 'Memory usage' in line:
            continue
        elif 's|' in line:
            try:
                what_idx = line.index('Built')+len('Built ')
            except ValueError:
                continue
            try:
                what_next = what_idx+line[what_idx:].index('.')-1
            except ValueError:
                print("\t", line[what_idx:])
                raise
            what = line[what_idx:what_next]
            cutoff = what_next+line[what_next:].index('s|')
            segment = what_next+line[what_next:cutoff].rindex('[')+1
            section = line[segment:cutoff]
        else:
            for maybe in suffix100s:
                if maybe in line:
                    what = maybe
                    # Believe it or not, the spacing is inconsistent so I'm just
                    # doing this to account for it; never heard of regex who's that
                    section = line.rstrip()
                    section = line[line.rindex('s')-1]
                    while line[-1] not in "0123456789":
                        line = line[:-1]
                    section = line[line.rindex(' ')+1:]
                    break
        print("\t", what, '---', section)
        try:
            total_time += float(section)
        except ValueError:
            print("\t", line)

    print(f"Total time for {fname}: {total_time}")

