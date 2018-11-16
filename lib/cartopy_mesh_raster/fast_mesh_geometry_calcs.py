"""
Numba-enabled calculations to search a structured 2D mesh for faces containing
random target points.

The code assumes that the mesh nodes nearest to the target points are available
as an extra input, as this makes the whole process more tractable.

"""
import numba
import numpy as np


@numba.jit(nopython=True)
def create_node_faces_array(face_nodes, num_nodes):
    """
    Invert the face-to-node relationship to get node_faces(n_nodes, 4).

    We need this result to be an array, for numba-izing.

    We assume that all points touch *at most* 4 faces :
    * For points touching < 4 faces in the data, we will get -1 indices.
    * For points touching > 4 faces, we will get an error.

    """
    vertex_faces = np.zeros((num_nodes, 4), dtype=np.int32) - 1
    big = vertex_faces[0, 0]
    for f_index in range(face_nodes.shape[0]):
        face = face_nodes[f_index]
        for vertex in face:
            # Put the face's vertex in the appropriate index of the
            # vertex_faces array
            for i in range(4):
                if vertex_faces[vertex, i] == big:
                    vertex_faces[vertex, i] = f_index
                    # Don't look for any more
                    break
            else:
                raise RuntimeError('')

    return vertex_faces


@numba.njit()
def cross_product(a1, a2):
    """
    Calculate the cross product of matched arrays of points.

    Input shapes are [points-dim-a, points-dim-b ... 3[=xyz]]
    Output is same.

    """
    result = np.empty(a1.shape)
    result[..., 0] = a1[..., 1] * a2[..., 2] - a1[..., 2] * a2[..., 1]
    result[..., 1] = a1[..., 2] * a2[..., 0] - a1[..., 0] * a2[..., 2]
    result[..., 2] = a1[..., 0] * a2[..., 1] - a1[..., 1] * a2[..., 0]
    return result


@numba.njit()
def dot_product(a1_xyz, a2_xyz):
    """
    Calculate the dot product of matched arrays of points.

    Input shapes are [dim-a, dim-b .. dim_z, 3[=xyz]]
    Output shape is [dim-a, dim-b .. dim_z]

    """
    return np.sum(a1_xyz * a2_xyz, axis=-1)


@numba.njit()
def face_edge_normals(faces_points):
    """
    Calculate array of face edge normals from an array of face points.

    Input is [faces-dim-a, faces-dim-b ..., N_MESH_PTS_PER_FACE, 3[=xyz]]
    Output is [faces-dim-a, faces-dim-b ..., N_MESH_EDGES_PER_FACE, 3[=xyz]]

    """
    faces_next_points = np.concatenate((faces_points[:, 1:2, :],
                                        faces_points[:, 2:3, :],
                                        faces_points[:, 3:4, :],
                                        faces_points[:, 0:1, :]), axis=1)
    result = cross_product(faces_next_points, faces_points)
    return result


@numba.njit()
def search_faces_for_point(target_point_xyz,
                           i_point_nearest=-1,
                           face_indices_to_search=None,
                           mesh_points_xyz_array=None,
                           mesh_face_points_array=None,
                           mesh_point_faces_array=None):
    """
    Search a group of faces to see which contains a point.

    Args:

    * target_point_xyz (array[3]):
        xyz point target of face-search.

    Kwargs:

    * i_point_nearest (int):
        node index of the nearest mesh point.
        Only the faces connected to this node are searched.
        Ignored if face_indices_to_search is set.

    * face_indices_to_search (iterable of int):
        face indices of faces to search.
        Not required if i_point_nearest is a valid point index.

    * mesh_points_xyz_array (array[N_NODES, 3] of float):
        the 3d XYZ coordinates of the mesh nodes on the unit sphere.

    * mesh_face_points_array (array[N_FACES, 4] of int):
        the node indices of the (4) corners comprising each face.

    * mesh_point_faces_array (array[N_NODES, 4] of int):
        indices of the faces of which each point is a corner.
        Where a point touches < 4 faces, the unused face indices should be -1.

    Returns:
        n_found, face_index (int, int):
            (1, face-index), if point is contained in just one search face.
            (n, first-found-face-index) if in more than one of them
                i.e. n > 1 :  This may occur when close to an edge.
            (0, -1) if not in any of them.

    .. note::
        the last 3 kwargs are in fact required, and *not* optional.
        This is just for a nicer arg ordering.

    """
    # Get indices of required faces: [n_faces]
    if face_indices_to_search is None:
        face_indices_to_search = mesh_point_faces_array[i_point_nearest]

    # Extract face-points-indices [n_faces, N_MESH_PTS_PER_FACE]
#    faces_points_indices = mesh_face_points_array[
#        tuple(face_indices_to_search), :]
    n_search_faces = face_indices_to_search.size
    n_points_per_face = mesh_face_points_array.shape[-1]
    faces_points_indices = np.empty(
        (n_search_faces, n_points_per_face),
        dtype=np.int32)
    for ind, i_face in enumerate(face_indices_to_search):
        faces_points_indices[ind, :] = mesh_face_points_array[i_face, :]

    # Extract face-points-xyz [n_faces, N_MESH_PTS_PER_FACE, 3]
#    faces_points_xyz = _MESH_POINTS_XYZ[tuple(faces_points_indices), :]
    faces_points_xyz = np.empty((n_search_faces, n_points_per_face, 3),
                                dtype=mesh_points_xyz_array.dtype)
    for i_face in range(n_search_faces):
        for i_pt in range(n_points_per_face):
            faces_points_xyz[i_face, i_pt] = \
                mesh_points_xyz_array[faces_points_indices[i_face, i_pt]]

    # Calculate face-edge-normals [n_faces, N_MESH_EDGES_PER_FACE, 3]
    edge_normals = face_edge_normals(faces_points_xyz)
    # Calculate (target * edges) dot products [n_faces, N_MESH_EDGES_PER_FACE]
    point_edge_distances = dot_product(
        edge_normals,
        target_point_xyz.reshape((1, 1, 3)))
    # Get faces where 'inside' all 4 edges [n_faces].
    point_inside_edges = point_edge_distances <= 0.0
#    point_in_faces = np.prod(point_inside_edges, axis=-1)
    point_in_faces = point_inside_edges[..., 0]
    n_edges = point_inside_edges.shape[-1]
    for i_edge in range(1, n_edges):
        point_in_faces = np.logical_and(point_in_faces,
                                        point_inside_edges[..., i_edge])

    # Return number of hits + index of first (or -1).
    n_found = np.sum(point_in_faces)
    if n_found > 0:
        i_first = np.argmax(point_in_faces)
        face_index = face_indices_to_search[i_first]
    else:
        face_index = -1
    return (n_found, face_index)


@numba.njit()
def search_faces_for_points(target_points_xyz,
                            i_points_nearest,
                            nodes_xyz,
                            face_nodes_array,
                            node_faces_array):
    """
    Find the faces that contain target points.

    Args:

    * target_points_xyz (array[N_TARGET, 3] of float):
        the xyz target points (on the unit sphere).

    * i_points_nearest (array(N_TARGET) of int):
        node indexes of nearest mesh point, for each target point.
        For each target, only the faces connected to this node are searched.

    * mesh_points_xyz_array (array[N_NODES, 3] of float):
        the 3d XYZ coordinates of the mesh nodes on the unit sphere.

    * mesh_face_points_array (array[N_FACES, 4] of int):
        the node indices of the (4) corners of each face.

    * mesh_point_faces_array (array[N_NODES, 4] of int):
        indices of the faces of which each point is a corner.
        Where a point touches < 4 faces, the unused face indices should be -1.

    Returns:
        n_found (array[N_TARGET] of int):
            face indices of the faces containing each target point.

    """
    n_points = target_points_xyz.shape[0]
    result = np.zeros((n_points,), dtype=np.int32)
    for i_point in numba.prange(n_points):
        target_point_xyz = target_points_xyz[i_point]
        i_point_nearest = i_points_nearest[i_point]
        n_found, face_index = search_faces_for_point(
            target_point_xyz=target_point_xyz,
            i_point_nearest=i_point_nearest,
            mesh_points_xyz_array=nodes_xyz,
            mesh_point_faces_array=node_faces_array,
            mesh_face_points_array=face_nodes_array)
        result[i_point] = face_index
    return result
