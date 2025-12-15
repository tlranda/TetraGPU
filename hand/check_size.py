import meshio
import numpy as np
import sys
import subprocess

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

def check_size(fname):
    try:
        mesh = meshio.read(fname)
    except:
        # What you're catching here is sys.exit() because meshio is programmed by clowns -- the error is only logged to stdout so we pray it hits a match on version 2.2
        print(f"Meshio fails to read due to version being unsupported. Manually re-writing file to version 0.1 (JUST version # change, no other data/format changes)")
        version_str = get_version(fname)
        command = ["sed", f"0,/{version_str}/"+"{s/"+version_str+"/0.1/}", "-i", fname]
        subprocess.run(command)
        try:
            mesh = meshio.read(fname)
        except:
            print(f"Still failed to read")
            raise
    n_points = len(mesh.points)
    print(f"Mesh {fname} has {n_points} points (8 partitions expected @ {n_points//2} points)")
    print(" ".join(sorted(mesh.point_data.keys())))
    try:
        index = mesh.point_data["_index"]
    except KeyError:
        print(f"Mesh {fname} does NOT have an '_index' partition")
        return
    n_partitions = len(np.unique(index))
    print(f"Mesh {fname} has {n_partitions} unique partitions")

if __name__ == "__main__":
    if len(sys.argv) < 2 or "-h" in sys.argv or "--help" in sys.argv:
        print(f"USAGE: python3 {sys.argv[0]} <FILES_TO_CHECK>")
        print(f"Files are usually .vtu meshes")
        exit(0)
    for fname in sys.argv[1:]:
        check_size(fname)

