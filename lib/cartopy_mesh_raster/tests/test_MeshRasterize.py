"""
Tests for cartopy_mesh_raster.MeshRasterize.

"""
import os.path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

import cartopy.crs as ccrs

from cartopy_mesh_raster import MeshRasterize


def test_plot_cubesphere():
    import cartopy_mesh_raster.tests as cmrt
    test_filename = 'tiny_4x4_cubesphere.nc'
    test_filepath = os.sep.join([os.path.dirname(cmrt.__file__),
                                'test_data',
                                test_filename])

    with nc.Dataset(test_filepath, 'r') as ds:
        latlon_coords_varnames = ('Mesh2_node_y', 'Mesh2_node_x')
        var_node_lats, var_node_lons = (
            ds.variables[varname]
            for varname in latlon_coords_varnames)
        node_lats, node_lons = (
            var[:]
            for var in (var_node_lats, var_node_lons))

        units = var_node_lons.units
        if 'degree' not in units:
            node_lats, node_lons = (np.rad2deg(arr)
                                    for arr in (node_lats, node_lons))

        face_nodes_indices = ds.variables['Mesh2_face_nodes'][:] - 1

        face_values = ds.variables['face_values'][:]

    raster = MeshRasterize(lons=node_lons, lats=node_lats,
                           face_nodes=face_nodes_indices,
                           img=face_values)

    ax = plt.axes(xlim=[-180, 180], ylim=[-90, 90],
                  projection=ccrs.PlateCarree())
    min_val, max_val = (np.min(face_values), np.max(face_values))
    ax.add_raster(raster, norm=plt.Normalize(min_val, max_val),
                  origin='lower')
    ax.coastlines()
    plt.show()


if __name__ == '__main__':
    test_plot_cubesphere()
