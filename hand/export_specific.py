import meshio
import numpy as np

import argparse
import itertools
import pathlib
import subprocess
import sys

def get_version(fname):
    proc = subprocess.run(["head", "-n2", fname], capture_output=True)
    headstr = proc.stdout.decode('utf-8')
    # Tail to second line
    headstr = headstr.split('\n')[1]
    # Isolate version=
    headstr = headstr[headstr.find('version'):].split(' ',1)[0]
    # Get version
    headstr = headstr[headstr.find('"')+1:headstr.rfind('"')]
    return headstr

def export_specific(args):
    try:
        mesh = meshio.read(args.mesh)
    except:
        # What you're catching here is sys.exit() because meshio is programmed by clowns -- the error is only logged to stdout so we pray it hits a match on version 2.2
        print(f"Meshio fails to read due to version being unsupported. Manually re-writing file to version 0.1 (JUST version # change, no other data/format changes)")
        version_str = get_version(args.mesh)
        command = ["sed", f"0,/{version_str}/"+"{s/"+version_str+"/0.1/}", "-i", args.mesh]
        subprocess.run(command)
        try:
            mesh = meshio.read(args.mesh)
        except:
            print(f"Still failed to read")
            raise
    if args.list_arrays:
        print(f"Found {len(mesh.point_data)} point arrays")
        print("\t-"+"\n\t-".join(mesh.point_data.keys()))
        print(f"Found {len(mesh.cell_data)} cell arrays")
        print("\t-"+"\n\t-".join(mesh.cell_data.keys()))
        return
    saveable_point_data = {}
    for key in args.keep_arrays:
        # Possibility of KeyError intended to save you from dumbassery
        saveable_point_data[key] = mesh.point_data[key]
    mesh.point_data = saveable_point_data
    # If requested, look up and add external cells to cell_data
    if args.add_external:
        if "_index" not in mesh.point_data:
            raise ValueError("No partitioning information '_index' array in mesh!")
        # Something like this PER partition in the mesh
        #mesh.cell_data[f'partition_{}_external'] = [0 if _ not in within_partition_points else 1 for _ in mesh.points]
        partitions = sorted(set(mesh.point_data['_index']))
        for (pid, partition) in enumerate(partitions):
            within_partition = np.where(mesh.point_data['_index'] == partition)[0]
            n_cells = len(mesh.cells[0])
            inclusion = np.zeros(n_cells)
            for cid, cell in enumerate(mesh.cells[0].data):
                if any((vertex in within_partition for vertex in cell)):
                    inclusion[cid] = 1
            mesh.cell_data[f'partition_cells_{pid}'] = [inclusion]
    mesh.write(args.export)

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("mesh", type=pathlib.Path, help="VTU file to process")
    prs.add_argument("--export", type=pathlib.Path, default=None, help="Name to output to (with only kept arrays)")
    prs.add_argument("--list-arrays", action="store_true", help="List arrays of the mesh")
    prs.add_argument("--keep-arrays", default=None, nargs="*", action="append", help="Names of arrays to keep")
    prs.add_argument("--add-external", action="store_true", help="Add external TV cell IDs per partition")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    if args.keep_arrays is None:
        if not args.list_arrays:
            raise ValueError(f"You do NOT want to delete EVERY array! Try listing the ones to remove first with --list")
    else:
        args.keep_arrays = list(itertools.chain.from_iterable(args.keep_arrays))
    if not args.list_arrays and args.export is None:
        raise ValueError(f"Must give --export target if not just using --list to view arrays")
    return args

if __name__ == "__main__":
    args = parse()
    export_specific(args)

