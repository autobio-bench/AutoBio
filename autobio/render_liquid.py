import pickle
from pathlib import Path

import numpy as np
import mujoco
import shapely
from pxr import Usd, UsdGeom
from numba import njit
from scipy.spatial import Delaunay

from serialize import load_log, take_state_split
from meshplane import Mesh, MeshPlane, make_plane_frame


def meniscus_1d(rho: float, g: float, gamma: float, theta: float):
    # https://users-phys.au.dk/srf/hydro/Landau+Lifschitz.pdf
    # http://brennen.caltech.edu/fluidbook/fluidstatics/meniscus.pdf
    # Capillary constant
    a = (2 * gamma / rho / g) ** 0.5
    z_high = a * np.sqrt(1 - np.sin(theta))

    if np.isclose(z_high, 0):
        return np.array([0, 0.1]), np.array([0, 0])

    sqrt_2 = np.sqrt(2)

    z = np.linspace(z_high, z_high * 0.01, 1000)
    x = -a / sqrt_2 * np.acosh(sqrt_2 * a / z) + a * np.sqrt(2 - z**2 / a**2)
    x -= x[0]
    if theta > np.pi / 2:
        z = -z
    return -x, z

def compute_meniscus(
    vertices: np.ndarray, faces: np.ndarray, boundary: np.ndarray, vertex_normals: np.ndarray, surface_normal: np.ndarray, surface_distance: float,
    *,
    rho: float = 1000.0,  # kg/m^3
    g: float = 9.81,  # m/s^2
    gamma: float = 71.97e-3,  # N/m
    theta: float = 70 / 180 * np.pi,  # contact angle in radians
):
    surface_frame = make_plane_frame(surface_normal)

    boundary_vertices = vertices[boundary]
    boundary_normals = vertex_normals[boundary]

    boundary_vertices_planar = boundary_vertices @ surface_frame
    assert np.allclose(boundary_vertices_planar[:, 2], surface_distance), breakpoint()
    boundary_vertices_planar = boundary_vertices_planar[:, :2]
    boundary_normals_planar = boundary_normals @ surface_frame

    # separate contact angle works by itself, but cause difficulty in merging with boundary
    # so use chosen_theta instead of real_theta
    real_theta = np.pi / 2 - np.acos(boundary_normals_planar[:, 2]) + theta
    if theta > np.pi / 2:
        # hydrophobic
        chosen_theta = np.maximum(real_theta.min(), np.pi / 2)
    else:
        # hydrophilic
        chosen_theta = np.minimum(real_theta.max(), np.pi / 2)

    xref, zref = meniscus_1d(rho, g, gamma, chosen_theta)
    boundary_z = zref[0]

    x = np.linspace(boundary_vertices_planar[:, 0].min(), boundary_vertices_planar[:, 0].max(), 20)
    y = np.linspace(boundary_vertices_planar[:, 1].min(), boundary_vertices_planar[:, 1].max(), 20)
    X, Y = np.meshgrid(x, y)

    polygon = shapely.Polygon(boundary_vertices_planar)
    shapely.prepare(polygon)

    contain_mask = shapely.contains_xy(polygon, X, Y)
    X = X[contain_mask]
    Y = Y[contain_mask]
    P = np.stack((X, Y), axis=-1)

    W, D = mean_value_coordinate(P, boundary_vertices_planar)

    Z = np.sum(W * np.interp(D, xref, zref), axis=1) - boundary_z

    Vxy = np.concatenate((boundary_vertices_planar, P), axis=0)
    Vz = np.concatenate((np.zeros(len(boundary_vertices_planar)), Z), axis=0)
    Vz += surface_distance
    V = np.concatenate((Vxy, Vz[:, None]), axis=1)
    new_faces = Delaunay(Vxy, qhull_options='QJ').simplices

    all_vertices, all_faces = merge_meniscus(
        vertices, faces, boundary, V, new_faces, surface_frame
    )
    return all_vertices, all_faces

def merge_meniscus(body_vertices: np.ndarray, body_faces: np.ndarray, boundary: np.ndarray, surface_vertices: np.ndarray, surface_faces: np.ndarray, surface_frame: np.ndarray):
    surface_vertices = surface_vertices @ surface_frame.T
    surface_faces = surface_faces.copy()
    nboundary = len(boundary)
    assert np.allclose(body_vertices[boundary], surface_vertices[:nboundary])
    new_vertices = surface_vertices[nboundary:]
    boundary_mask = surface_faces < nboundary
    surface_mask = surface_faces >= nboundary
    surface_faces[boundary_mask] = boundary[surface_faces[boundary_mask]]
    surface_faces[surface_mask] += len(body_vertices) - nboundary
    all_vertices = np.concatenate([body_vertices, new_vertices], axis=0)
    all_faces = np.concatenate([body_faces, surface_faces], axis=0)
    return all_vertices, all_faces

@njit(cache=True)
def mean_value_coordinate(points: np.ndarray, vertices: np.ndarray):
    # mean-value coordinates
    # https://cgvr.informatik.uni-bremen.de/teaching/cg_literatur/barycentric_floater.pdf

    def roll_batch(a, shift):
        b = np.empty_like(a)
        b[shift:] = a[:-shift]
        b[:shift] = a[-shift:]
        return b
    
    def norm_sample(v):
        r = np.zeros(len(v))
        for i in range(len(v)):
            r[i] = np.linalg.norm(v[i])
        return r

    post_edges = roll_batch(vertices, -1) - vertices
    pre_edges = roll_batch(vertices, 1) - vertices
    post_direction = post_edges / norm_sample(post_edges)[:, None]
    pre_direction = pre_edges / norm_sample(pre_edges)[:, None]

    w = np.zeros((len(points), len(vertices)))
    d = np.zeros((len(points), len(vertices)))
    for i, p in enumerate(points):
        vector = vertices - p
        distance = norm_sample(vector)
        coincident = np.isclose(distance, 0)
        coincidence_indices = np.where(coincident)[0]
        if len(coincidence_indices) > 1:
            raise ValueError("More than one coincident vertex")
        elif len(coincidence_indices) == 1:
            index = coincidence_indices[0]
            w[i, index] = 1
        else:
            direction = vector / distance[:, None]
            direction2 = roll_batch(direction, -1)
            cos_alpha = np.sum(direction * direction2, axis=1)
            colinear = np.isclose(cos_alpha, -1)
            colinear_indices = np.where(colinear)[0]
            if len(colinear_indices) > 1:
                raise ValueError("More than one colinear edge")
            elif len(colinear_indices) == 1:
                index1 = colinear_indices[0]
                index2 = (index1 + 1) % len(vertices)
                # linear interpolation
                distance1 = distance[index1]
                distance2 = distance[index2]
                w[i, index1] = distance2 / (distance1 + distance2)
                w[i, index2] = distance1 / (distance1 + distance2)
            else:
                sin_alpha = direction[:, 0] * direction2[:, 1] - direction[:, 1] * direction2[:, 0]
                alpha = np.arctan2(sin_alpha, cos_alpha)
                tan_alpha_half = np.tan(alpha / 2)
                tan_alpha_half_prev = np.roll(tan_alpha_half, 1)
                w[i] = (tan_alpha_half_prev + tan_alpha_half) / distance
                w[i] /= np.sum(w[i])
        
        side_post = np.sum(vector * post_direction, axis=1)
        side_pre = np.sum(vector * pre_direction, axis=1)
        side = np.maximum(np.abs(side_post), np.abs(side_pre))
        d[i] = np.sqrt(np.maximum(0, distance**2 - side**2))

    return w, d

def render_liquid(log_dir: Path, fps: int = 20, meniscus: bool = False):
    log_dir = Path(log_dir)
    model, states, info = load_log(log_dir)
    with open(log_dir / "liquid.pkl", "rb") as f:
        liquids = pickle.load(f)
    
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0)
    stage.SetTimeCodesPerSecond(fps)

    qpos = take_state_split(states, info['split']['qpos'])
    timestep = model.opt.timestep
    num_steps = qpos.shape[0]
    step = 1 / fps / timestep
    if not np.isclose(step, int(step)):
        print(f"Warning: Inexact step size {step} for timestep {timestep} and fps {fps}")
    indices = np.arange(0, num_steps, step)
    indices = np.rint(indices).astype(int)
    qpos = qpos[indices]
    num_frames = len(indices)

    for liquid in liquids:
        mesh = Mesh(liquid['vertices'], liquid['faces'].astype(np.uint64), liquid['boundary'])
        meshplane = MeshPlane(mesh)

        normals = liquid['normal'][indices]
        distances = liquid['distance'][indices]

        usd_mesh = UsdGeom.Mesh.Define(stage, f"/liquid_{liquid['geom_id']}")
        vertices_attr = usd_mesh.GetPointsAttr()
        face_vertex_counts_attr = usd_mesh.GetFaceVertexCountsAttr()
        face_vertex_indices_attr = usd_mesh.GetFaceVertexIndicesAttr()

        for i in range(num_frames):
            meshplane.set_plane_normal(*normals[i])
            result_mesh = meshplane.calculate_mesh(distances[i])
            
            if len(result_mesh.boundary) > 0:
                if meniscus:
                    vertices, faces = compute_meniscus(
                        result_mesh.vertices, result_mesh.faces, result_mesh.boundary, result_mesh.vertex_normals, normals[i], distances[i]
                    )
                    vertices_attr.Set(vertices, time=i)
                    vertex_counts = np.full(len(faces), 3, dtype=np.int32)
                    face_vertex_counts_attr.Set(vertex_counts, time=i)
                    face_vertex_indices_attr.Set(faces, time=i)
                else:
                    vertices_attr.Set(result_mesh.vertices, time=i)
                    vertex_counts = np.empty(len(result_mesh.faces) + 1, dtype=np.int32)
                    vertex_counts[:-1] = 3
                    vertex_counts[-1] = len(result_mesh.boundary)
                    face_vertex_counts_attr.Set(vertex_counts, time=i)
                    faces = np.concatenate((result_mesh.faces.flatten(), result_mesh.boundary))
                    face_vertex_indices_attr.Set(faces, time=i)
            else:
                vertices_attr.Set(result_mesh.vertices, time=i)
                vertex_counts = np.full(len(result_mesh.faces), 3, dtype=np.int32)
                face_vertex_counts_attr.Set(vertex_counts, time=i)
                face_vertex_indices_attr.Set(result_mesh.faces, time=i)
    stage.GetRootLayer().Export(str(log_dir / "liquid.usd"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path, help="Directory of the log file")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--meniscus", action="store_true", help="Render meniscus")
    args = parser.parse_args()

    mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')
    render_liquid(args.log_dir, fps=args.fps, meniscus=args.meniscus)
