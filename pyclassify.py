import numpy as np

import argparse
import pathlib

class ConstInt:
    def __init__(self, imin=None, imax=None):
        self.imin = imin
        self.imax = imax
    def __call__(self, arg):
        try:
            value = int(arg)
        except ValueError:
            raise self.exception(f"Could not parse as integer")
        if (self.imin is not None and value < self.imin):
            raise self.exception(f"Minimum accepted value is {self.imin}")
        if (self.imax is not None and value > self.imax):
            raise self.exception(f"Maximum accepted value is {self.imax}")
        return value
    def exception(self, reason):
        return argparse.ArgumentTypeError(reason)

dhelp = "(Default: %(default)s)"
prs = argparse.ArgumentParser()
prs.add_argument('target_file', type=pathlib.Path, help="Output from my critical points script")
prs.add_argument('--verify', type=pathlib.Path, default=None, help=f"Paraview-converted results to verify against {dhelp}")
prs.add_argument('--max-display', type=ConstInt(imin=0), default=10, help=f"Maximum number of points to display {dhelp}")

args = prs.parse_args()

# Name mapping to class IDs
#              0   1     2     3         4
class_names = ['','min','max','regular','saddle']
insanity, classes = list(), dict((k,list()) for k in range(len(class_names)))
if args.verify is not None:
    match_paraview = [1,4,4,2,4,3]
    paraview_string = {'Minima': 0, '1-Saddle': 1, '2-Saddle': 2, 'Maxima': 3, 'Multi-Saddle': 4, 'Regular': 5}
    verify_points = list()
    current_point = 0
    with open(args.verify,'r') as f:
        for l in f.readlines():
            if len(l.split(' ')) != 2:
                raise ValueError(f"Incorrectly formatted verify file (needs to be <IDX, CLASS>, specify --full-vtk at conversion time to permit usage with this script!)")
            idx, classname = l.rstrip().split(' ')
            idx = int(idx)
            try:
                classname = match_paraview[int(classname)]
            except:
                # String typed
                classname = match_paraview[paraview_string[classname]]
            # Regular points are not usually logged
            verify_points.extend([3]*(idx-current_point))
            verify_points.append(classname)
            current_point = idx+1
    verify_points = np.asarray(verify_points)
    verify_lookup = dict()
    for idx, name in enumerate(class_names):
        verify_lookup[name] = np.where(verify_points == idx)[0]

with open(args.target_file,'r') as f:
    for l in f.readlines():
        # Longer form output detected
        if "Upper:" in l and "Lower:" in l:
            # Other output format
            vertex_id = int(l.split(' ',3)[2])
            for last_int, class_name in enumerate(class_names[1:]):
                if class_name in l:
                    classes[last_int+1].append(vertex_id)
                    print(f"Point {vertex_id} is {class_names[last_int+1]}")
                    break
            # Skip normal processing
            continue
        # Insanity or short form output detected
        last_int = int(l.rstrip().rsplit(' ',1)[1])
        if 'INSANITY' in l:
            insanity.append(last_int)
        else:
            vertex_id = int(l.rsplit(" ",3)[1])
            classes[last_int].append(vertex_id)
            print(f"Point {vertex_id} is {class_names[last_int]}")

print(f"{len(insanity)} insane points detected: {insanity[:args.max_display]}")
for name, clinfo in zip(class_names, classes.values()):
    if name == "":
        continue
    print(f"{len(clinfo)}{f'/{len(verify_lookup[name])}' if args.verify else ''} {name} points detected: {clinfo[:args.max_display]}")

if args.verify is not None:
    any_mia = False
    for classname in classes.keys():
        expect = verify_lookup[class_names[classname]]
        actual = np.asarray(classes[classname])
        mia = np.where(~np.in1d(expect,actual))[0]
        if len(mia) > 0:
            any_mia = True
            print(f"{len(mia)} misclassifications for {class_names[classname]}: {expect[mia[:args.max_display]]}")
    if not any_mia:
        print(f"Fully verified :)")
