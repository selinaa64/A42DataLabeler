from plane_estimation import estimate_plane
import open3d as o3d

def create_lane_box(extent, center):
    rotation_matrix = estimate_plane()
    return o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)