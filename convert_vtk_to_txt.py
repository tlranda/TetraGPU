import pyvista as pv
import numpy as np
import tqdm

import argparse
import pathlib

def convert_vtk_to_txt(vtk_file_path, txt_file_path, total_points=None, global_indices=None, class_as_str=False):
    # Load the dataset
    mesh = pv.read(vtk_file_path)

    if mesh.n_points == 0:
        raise ValueError(f"No points found in the VTK file '{vtk_file_path}'")

    # Get point coordinates
    points = mesh.points  # shape (n_points, 3)

    # Get scalar values (assuming there's only one scalar array)
    scalar_names = mesh.point_data.keys()
    if not scalar_names:
        raise ValueError(f"No scalar data found in VTK file '{vtk_file_path}'")

    print(f"Available arrays: {scalar_names}")
    scalar_names = list(scalar_names)
    if 'CriticalType' not in scalar_names:
        raise ValueError(f"'CriticalType' data is not included in '{vtk_file_path}'")
    scalar_name = scalar_names[scalar_names.index('CriticalType')]
    scalars = mesh.point_data[scalar_name]  # shape (n_points,)
    # Paraview Class ID #'s
    #                             0        1          2          3        4              5(unused)
    vtk_class_name = np.asarray(['Minima','1-Saddle','2-Saddle','Maxima','Multi-Saddle','Regular'])
    class_counts = dict((vtk_class_name[_],list(scalars).count(_)) for _ in sorted(set(scalars)))
    #print(vtk_class_name[scalars])
    if class_as_str:
        scalars = vtk_class_name[scalars]
    if 'Regular' not in class_counts.keys():
        if total_points is None:
            total_points = scalars.shape[0]
        class_counts['Regular'] = total_points-sum(class_counts.values())
    print(f"Class counts: {class_counts}")

    # Stack and save
    if global_indices is None:
        # Ensure different dtypes can be applied easily
        data = np.rec.fromarrays((points[:,0],points[:,1],points[:,2],scalars),
                                 names=('x','y','z','classname'))
        npfmt = ["%.6f"]*3+["%s" if class_as_str else "%d"]
    else:
        full_mesh = pv.read(global_indices)
        if full_mesh.n_points == 0:
            raise ValueError(f"No points found in VTK file '{global_indices}'")
        indices = list()
        for point in points:
            indices.append(full_mesh.FindPoint(point))
            #print(f"Point {point} located at {indices[-1]}")
        # Ensure different dtypes can be applied easily
        data = np.rec.fromarrays((indices,scalars), names=('index','classname'))
        npfmt = ["%d"]+["%s" if class_as_str else "%d"]
    np.savetxt(txt_file_path, data, fmt=npfmt)
    print(f"Conversion complete: {txt_file_path}")

prs = argparse.ArgumentParser()
prs.add_argument('input', type=pathlib.Path, help="VTK file for input, with saved critical points")
prs.add_argument('output', type=pathlib.Path, help="TXT file for output, with XYZ of critical point per line of output")
prs.add_argument('--total-points', type=int, default=None, help="Full size of the mesh for # regular points correction")
prs.add_argument('--full-vtk', type=pathlib.Path, default=None, help="VTK file with all points to identify the global index")
prs.add_argument('--class-as-string', action='store_true', help="Export classes as strings rather than integer enums")
args = prs.parse_args()
convert_vtk_to_txt(args.input, args.output, args.total_points, args.full_vtk, args.class_as_string)

