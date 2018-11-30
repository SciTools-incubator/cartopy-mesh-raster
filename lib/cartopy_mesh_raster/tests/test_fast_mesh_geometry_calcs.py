from datetime import datetime

import numpy as np

from cartopy_mesh_raster import fast_mesh_geometry_calcs as fmgc

#
# Map of test mesh :
#    points p1-p9
#    (lat,lon) locations
#    faces F1-F4.
#    "+" test target points
#
# p3(33, 3)-------p4(34, 54)-------p9(39, 99)
#      |               |                |
#      |      F=2      |      F=3       |
#      |             + | +              |
# p2(22, 2)-------p5(25, 55)-------p8(28, 98)
#      |             + | +              |
#      |      F=1      |      F=4       |
#      |               |                |
# p1(11, 1)-------p6(16, 56)-------p7(17, 97)
#

# 6 testcases --> target-point(lat, lon), PLUS starting("nearest") mesh-point.
# Note: the 4 testpoints are all *really* nearest to p5 (== index 4 below).
# Using startpoint=3 (p4) should produces the predictable errors.
test_points = [
    ([23., 53], 4),
    ([27., 53], 4),
    ([24., 57], 4),
    ([27., 57], 4),
    ([25., 55], 4),
    ([23., 53], 3),
    ([27., 53], 3),
    ([24., 57], 3),
    ([27., 57], 3),
    ([25., 55], 3),
]

# Mesh point (aka NODE) locations.
MESH_POINT_LATLONS_DEG = np.array(
    [
     (11., 1),
     (22, 2),
     (33, 3),
     (34, 54),
     (25, 55),
     (16, 56),
     (17, 97),
     (28, 98),
     (39, 99),
    ])

# Mesh faces defined in terms of nodes.
MESH_FACE_POINTS = np.array(
    [
     (1, 6, 5, 2),
     (2, 5, 4, 3),
     (5, 8, 9, 4),
     (6, 7, 8, 5),
    ])
# index of 'point 1' is 0
# "-1" *should* mean missing (?)
MESH_FACE_POINTS -= 1


# Utility routine for converting lats+lons to xyz.
def _latlons_to_xyzs(lats, lons, in_degrees=False):
    if in_degrees:
        lats = np.deg2rad(lats)
        lons = np.deg2rad(lons)
    z = np.sin(lats)
    cos_lat = np.cos(lats)
    x = cos_lat * np.cos(lons)
    y = cos_lat * np.sin(lons)
    return np.stack((x, y, z), axis=-1)


def test():
    #
    # Note: no proper tests yet, just a method you can call.
    #

    # Construct mesh _arrays_ for the test mesh, as needed for calls.
    nodes_xyz = _latlons_to_xyzs(
        lats=MESH_POINT_LATLONS_DEG[:, 0],
        lons=MESH_POINT_LATLONS_DEG[:, 1],
        in_degrees=True)

    face_nodes = np.array(MESH_FACE_POINTS, dtype=np.int)

    node_faces = fmgc.create_node_faces_array(
        face_nodes, num_nodes=len(MESH_POINT_LATLONS_DEG))

    #
    # Exercise the single-point search call.
    #
    n_tests = len(test_points)
    all_testpoints_xyz = np.zeros((n_tests, 3), dtype=np.float)
    for i_test, (test_point, i_nearest) in enumerate(test_points):
        print('')
        print('target = {}, startpoint = {}'.format(test_point, i_nearest))
        target_latlon = np.array(test_point)
        target_xyz = _latlons_to_xyzs(*target_latlon, in_degrees=True)
        n_found, i_first = fmgc.search_faces_for_point(
            target_point_xyz=target_xyz,
            i_point_nearest=i_nearest,
            mesh_points_xyz_array=nodes_xyz,
            mesh_face_points_array=face_nodes,
            mesh_point_faces_array=node_faces
            )
        print('  result : n={}, i_face={}'.format(n_found, i_first))
        all_testpoints_xyz[i_test] = target_xyz

    #
    # Exercise multi-point search call.
    #
    test_nearest_points = [testcase[1] for testcase in test_points]
    nearest_indices_array = np.array(test_nearest_points, np.int)
    faces_found = fmgc.search_faces_for_points(
        target_points_xyz=all_testpoints_xyz,
        i_points_nearest=nearest_indices_array,
        nodes_xyz=nodes_xyz,
        face_nodes_array=face_nodes,
        node_faces_array=node_faces)

    print('')
    print('Multi-point result : {}'.format(faces_found))

    #
    # TEST : check that last result is as expected.
    #
    assert np.all(faces_found == [0, 1, 3, 2, 2, -1, 1, -1, 2, 2])

    #
    # Also do a simple speed test ..
    #
    print('')

    t0 = datetime.now()
    n_calls = 100
    for n in range(n_calls):
        faces_found = fmgc.search_faces_for_points(
            target_points_xyz=all_testpoints_xyz,
            i_points_nearest=nearest_indices_array,
            nodes_xyz=nodes_xyz,
            face_nodes_array=face_nodes,
            node_faces_array=node_faces)
    t1 = datetime.now()
    dt = (t1 - t0).total_seconds()

    print('Time per face : {}'.format(dt / (n_calls * n_tests)))


if __name__ == '__main__':
    test()
