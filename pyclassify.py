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
prs.add_argument('--n-verify-points', type=int, default=None, help=f"Number of points on the verifying mesh {dhelp}")
prs.add_argument('--autoscale-verify', action='store_true', help=f"Autoscale verification file size up to the size of the target file {dhelp}")
prs.add_argument('--max-display', type=ConstInt(imin=0), default=10, help=f"Maximum number of points to display {dhelp}")

args = prs.parse_args()

# Name mapping to class IDs
#              0       1     2     3         4
class_names = ['NULL','min','max','regular','saddle']
insanity, classes = list(), dict((k,list()) for k in range(len(class_names)))
with open(args.target_file,'r') as f:
    for l in f.readlines():
        # Longer form output detected
        if "Upper:" in l and "Lower:" in l:
            # Other output format
            vertex_id = int(l.split(' ',3)[2])
            for last_int, class_name in enumerate(class_names):
                if class_name in l:
                    classes[last_int].append(vertex_id)
                    #print(f"Point {vertex_id} is {class_names[last_int]}")
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
            #print(f"Point {vertex_id} is {class_names[last_int]}")

target_size = sum(map(len,classes.values()))

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
    if args.n_verify_points is not None:
        extension = args.n_verify_points - len(verify_points)
        if extension < 0:
            raise ValueError(f"Read more points ({len(verify_points)} points in '{args.verify}') than indicated mesh size ({args.n_verify_points})!")
        if extension > 0:
            verify_points.extend([3]*extension)
    elif args.autoscale_verify:
        extension = target_size - len(verify_points)
        verify_points.extend([3]*extension)
    verify_points = np.asarray(verify_points)
    verify_lookup = dict()
    for idx, name in enumerate(class_names):
        verify_lookup[name] = np.where(verify_points == idx)[0]

if args.n_verify_points is not None and args.n_verify_points != target_size:
    raise ValueError(f"Did not find definitions for all {args.n_verify_points} in '{args.target_file}' (found: {target_size} points)!")
elif args.verify is not None and target_size != len(verify_points):
    raise ValueError(f"Did not find same number of points between target '{args.target_file}' ({target_size}) and verification file '{args.verify}' ({len(verify_points)}); you may need to specify --autoscale-verify or --n-verify-points to fix this, or these files may not describe the same mesh!")

print(f"{len(insanity)} insane points detected: {insanity[:args.max_display]}")
for name, clinfo in zip(class_names, classes.values()):
    if name == "NULL":
        continue
    print(f"{len(clinfo)}{f'/{len(verify_lookup[name])}' if args.verify else ''} {name} points detected: {clinfo[:args.max_display]}")

if args.verify is not None:
    any_mia = False
    unclassified = len(classes[0]) > 0
    expect_noclass = np.asarray(classes[0])
    for classname in classes.keys():
        expect = verify_lookup[class_names[classname]]
        actual = np.asarray(classes[classname])
        mia = np.where(~np.in1d(expect,actual))[0]
        if unclassified:
            # Don't penalize points that we already expect to lack a class anyways
            if len(mia) > 0:
                # Prevent this from flagging as fully verified though!
                any_mia = True
            mia = np.where(~np.in1d(expect[mia],expect_noclass))[0]
        if len(mia) > 0:
            any_mia = True
            print(f"{len(mia)} misclassifications for {class_names[classname]}: {expect[mia[:args.max_display]]}")
    if not any_mia:
        print(f"Fully verified :)")
