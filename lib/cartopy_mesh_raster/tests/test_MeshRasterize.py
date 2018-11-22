"""
Tests for cartopy_mesh_raster.MeshRasterize.

"""
import iris.tests as tests

import os.path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

import cartopy.crs as ccrs
import cartopy_mesh_raster.tests as cmrt
from iris.tests import IrisTest

from cartopy_mesh_raster import MeshRasterize


class TestMeshRasterize(IrisTest):
    def setUp(self):
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

        self._face_values_min_max = (np.min(face_values), np.max(face_values))
        self._raster = MeshRasterize(lons=node_lons, lats=node_lats,
                                     face_nodes=face_nodes_indices,
                                     img=face_values)
        self._target_projection = ccrs.PlateCarree()

#    def test_fetch_raster__largescale(self):
#        extent = [0.0, 90.0, 0.0, 90.0]
#        target_dims_xy = (4, 3)
#        result, = self._raster.fetch_raster(
#            projection=self._target_projection,
#            extent=extent,
#            target_resolution=target_dims_xy)
#        result = result.image
#        expected = np.array([[5., 5., 5., 5.],
#                             [5., 3.75, 3.75, 1.],
#                             [4., 4., 4., 4.]])
#        self.assertArrayAllClose(result, expected)
#
#    def test_fetch_mini_raster__smallscale(self):
#        extent = [22.49, 22.51, 42.73, 42.74]
#        target_dims_xy = (4, 2)
#        result, = self._raster.fetch_raster(
#            projection=self._target_projection,
#            extent=extent,
#            target_resolution=target_dims_xy)
#        result = result.image
#        print(result)
#        expected = np.array([[-0.25, -0.25, 0., 0.],
#                             [3.75, 3.75, 3.75, 3.75]])
#        self.assertArrayAllClose(result, expected)
#
#    def test_ok_bad_extent(self):
#        extent = [-300, 300, -200, 200]
#        target_dims_xy = (4, 2)
#        result, = self._raster.fetch_raster(
#            projection=self._target_projection,
#            extent=extent,
#            target_resolution=target_dims_xy)

    def test_centre_y_points(self):
        extent = [-180, 180, -0.1, 0.1]
        target_dims_xy = (7, 5)
        result, = self._raster.fetch_raster(
            projection=self._target_projection,
            extent=extent,
            target_resolution=target_dims_xy)
        print(result)

#    def test_plot_raster(self):
#        plt.switch_backend('tkagg')
#        ax = plt.axes(xlim=[-180, 180], ylim=[-90, 90],
#                      projection=self._target_projection)
#        min_val, max_val = self._face_values_min_max
#        ax.add_raster(self._raster,
#                      norm=plt.Normalize(min_val, max_val),
#                      origin='lower')
#        ax.coastlines()
#        plt.show()


if __name__ == '__main__':
    tests.main()
