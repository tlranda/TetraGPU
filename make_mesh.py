# Dependent modules
import meshio
import numpy as np
import tqdm
# Builtin modules
import argparse
from itertools import product
import pathlib
import subprocess

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--n-points', type=int, default=5, help="Number of points in the mesh (each point beyond 4th adds +1 tetra) (Default: %(default)s)")
    prs.add_argument('--output', type=pathlib.Path, default='generated', help="Output VTU file for the mesh (Default: %(default)s)")
    prs.add_argument('--seed', type=int, default=1234, help="Numpy random seed (Default: %(default)s)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    if args.n_points < 4:
        raise ValueError("Must define at least 4 points")
    args.output = pathlib.Path(args.output).with_suffix('.vtu')
    np.random.seed(args.seed)
    return args

def make_3d_points(n_points):
    # Gives unique X-Y-Z combination of values in increments of 1.0 while
    # rotating through axes X->Z ish
    nrange = np.ceil(np.power(n_points,1/3)).astype(int)
    points = sorted(list(product(*([range(nrange)]*3)))[:n_points],
                    key=lambda x: (sum(x),x[2],x[1],x[0]))
    return points

def make_tetras(n_points):
    # First tetra is just the first four points and defines 4 connectable faces
    # to start us out
    tetras = [[0,1,2,3]]
    unused = list(set(range(n_points)).difference({0,1,2,3}))
    available_faces = {(0,1,2),(1,2,3),(0,2,3),(0,1,3)}

    # Every tetra afterwards needs to incorporate at least one unused point,
    # however faces can only be connected to TWO tetras at most
    for _ in tqdm.tqdm(range(len(unused)), total=len(unused)):
        # Pick an unused point to add to the mesh
        new_point = unused.pop()
        # Pick a face that isn't doubly-used to connect to the point
        reused_face = available_faces.pop()
        # Add this tetra to the mesh
        new_tetra = sorted(list(reused_face)+[new_point])
        tetras.append(new_tetra)
        # Track new faces added to the mesh by this tetra
        new_faces = {tuple(sorted([reused_face[0],reused_face[1], new_point])),
                     tuple(sorted([reused_face[1],reused_face[2], new_point])),
                     tuple(sorted([reused_face[0],reused_face[2], new_point])),}
        available_faces = available_faces.union(new_faces)
    # Return in meshio format
    return [('tetra', tetras)]


def main():
    args = parse()

    points = make_3d_points(args.n_points)
    tetras = make_tetras(args.n_points)

    scalar_field = np.random.rand(args.n_points)

    # Use Meshio to write in VTK's VTU format -- but use ASCII instead of binary
    # Technically I think you should be able to not use the subprocess below
    # but I am unsure of how to get their API to cooperate -- this works but
    # is probably a bit redundant and may slow down processing for some larger
    # meshes
    meshio.write_points_cells(args.output, points, tetras, point_data={"scalar_field": scalar_field})
    subprocess.run(['meshio', 'ascii', str(args.output)])
    # Unfortunately I don't see how to get this right initially, so we have to
    # fix the scalar ID on the point data after it gets exported
    subprocess.run(['sed', '-i', '-e', 's/<PointData>/<PointData Scalars="scalar_field">/', str(args.output)])

if __name__ == '__main__':
    main()

