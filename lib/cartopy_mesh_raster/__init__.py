import logging
import time

import cartopy.crs as ccrs
from cartopy.io import RasterSource, LocatedImage
import numpy as np
try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree

import cartopy_mesh_raster.fast_mesh_geometry_calcs as fmgc


class MeshRasterize(RasterSource):
    def __init__(self, lons, lats, face_nodes, img):
        """
        lons, lats : 1d array in node index order
        face_nodes: iterable of iterable of node index
        img: 1d array of source data. Same length as face nodes

        """
        # convert to 3d space
        self._geocent = ccrs.Geocentric(globe=ccrs.Globe())
        self.img = img
        xyz = self._geocent.transform_points(ccrs.Geodetic(), lons, lats)
        self._nodes_xyz = xyz
        start = time.time()
        self._kd = KDTree(xyz)
        end = time.time()
        logging.info('KD Construction time ({} points): {}'.format(
            lons.size, end - start))
        self._face_nodes = np.array(face_nodes)
        self._node_faces = fmgc.create_node_faces_array(self._face_nodes,
                                                        num_nodes=len(lons))

    def validate_projection(self, projection):
        return True

    def fetch_raster(self, projection, extent, target_resolution):
        target_resolution = np.array(target_resolution, dtype=int)
        x = np.linspace(extent[0], extent[1], target_resolution[0])
        y = np.linspace(extent[2], extent[3], target_resolution[1])
        xs, ys = np.meshgrid(x, y)
        xyz_sample = self._geocent.transform_points(
                projection, xs.flatten(), ys.flatten())
        start = time.time()
        _, node_indices = self._kd.query(xyz_sample, k=1)
        end = time.time()
        logging.info('Query of {} points: {}'.format(
            np.prod(target_resolution), end - start))

        # Clip to valid node indices: can get =N-points for NaN or inf. points.
        n_points = self._node_faces.shape[0]
        node_indices[(node_indices < 0) | (node_indices >= n_points)] = 0

        start = time.time()
        face_indices = fmgc.search_faces_for_points(
            target_points_xyz=xyz_sample,
            i_nodes_nearest=node_indices,
            nodes_xyz_array=self._nodes_xyz,
            face_nodes_array=self._face_nodes,
            node_faces_array=self._node_faces
            )
        end = time.time()
        logging.info('Face search of {} points: {}'.format(
            np.prod(target_resolution), end - start))

        result = self.img[face_indices].reshape(target_resolution[1],
                                                target_resolution[0])
        return [LocatedImage(result, extent)]
