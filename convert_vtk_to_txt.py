import pyvista as pv
import numpy as np
import tqdm

import argparse
import pathlib

def convert_vtk_to_txt(vtk_file_path, txt_file_path, total_points=None, global_indices=None):
    # Load the dataset
    mesh = pv.read(vtk_file_path)

    if mesh.n_points == 0:
        raise ValueError(f"No points found in the VTK file '{vtk_file_path}'")

    # Get point coordinates
    points = mesh.points  # shape (n_points, 3)

    # Get scalar values (assuming there's only one scalar array)
    scalar_names = mesh.point_data.keys()
    if not scalar_names:
        raise ValueError("No scalar data found in VTK file '{vtk_file_path}'")

    print(scalar_names)
    scalar_name = list(scalar_names)[0]
    scalars = mesh.point_data[scalar_name]  # shape (n_points,)
    vtk_class_name = ['Minima','1-Saddle','2-Saddle','Multi-Saddle','Maxima','Regular']
    class_counts = dict((vtk_class_name[_],list(scalars).count(_)) for _ in sorted(set(scalars)))
    if 'Regular' not in class_counts.keys():
        if total_points is None:
            total_points = scalars.shape[0]
        class_counts['Regular'] = total_points-sum(class_counts.values())
    print(f"Class counts: {class_counts}")

    # Stack and save
    if global_indices is None:
        data = np.hstack([points, scalars[:, np.newaxis]])
        npfmt = "%.6f"
    else:
        full_mesh = pv.read(global_indices)
        if full_mesh.n_points == 0:
            raise ValueError(f"No points found in VTK file '{global_indices}'")
        indices = list()
        for point in points:
            indices.append(full_mesh.FindPoint(point))
        data = np.vstack([indices, scalars]).T
        npfmt = "%d"
    np.savetxt(txt_file_path, data, fmt=npfmt)
    print(f"Conversion complete: {txt_file_path}")

prs = argparse.ArgumentParser()
prs.add_argument('input', type=pathlib.Path, help="VTK file for input, with saved critical points")
prs.add_argument('output', type=pathlib.Path, help="TXT file for output, with XYZ of critical point per line of output")
prs.add_argument('--total-points', type=int, default=None, help="Full size of the mesh for # regular points correction")
prs.add_argument('--full-vtk', type=pathlib.Path, default=None, help="VTK file with all points to identify the global index")
args = prs.parse_args()
convert_vtk_to_txt(args.input, args.output, args.total_points, args.full_vtk)

