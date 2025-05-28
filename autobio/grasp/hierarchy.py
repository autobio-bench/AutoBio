from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import sympy as sp

from grasp.body import Body, build_body
from grasp.equality import JointEquality, build_equality
from grasp.geom import Geom
from grasp.joint import Joint
from grasp.transform import Transform
from grasp.symbolic import K

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

def unit_slice(s: slice):
    assert s.stop - s.start == 1 and (s.step is None or s.step == 1)
    return s.start

@dataclass
class Hierarchy:
    bodies: dict[int, Body]
    equalities: list[JointEquality]
    root: int
    qs: slice

    @property
    def nbody(self):
        return len(self.bodies)

    @cached_property
    def nq(self):
        return self.qs.stop - self.qs.start

    @cached_property
    def ngeom(self):
        return sum(len(body.geoms) for body in self.bodies.values())

    @cached_property
    def njoint(self):
        return sum(len(body.joints) for body in self.bodies.values())

    @cached_property
    def geoms(self):
        geoms: dict[int, Geom] = {}
        for body in self.bodies.values():
            for geom in body.geoms:
                geoms[geom.id] = geom
        return geoms

    @cached_property
    def sites(self):
        sites: dict[int, Geom] = {}
        for body in self.bodies.values():
            for site in body.sites:
                sites[site.id] = site
        return sites

    @cached_property
    def joints(self):
        joints: dict[int, Joint] = {}
        for body in self.bodies.values():
            for joint in body.joints:
                joints[joint.id] = joint
        return joints

    @cached_property
    def joint_slices(self):
        slices = {}
        cursor = 0
        for joint in self.joints.values():
            slices[joint.id] = slice(cursor, cursor + joint.nq)
            cursor += joint.nq
        return slices

    @cached_property
    def bounds(self):
        return np.array(sum((joint.bound for joint in self.joints.values()), []))

    @cached_property
    def body_name2id(self):
        body_names = {body.name for body in self.bodies.values()}
        if len(body_names) < len(self.bodies):
            raise ValueError('Duplicate body name')
        return {body.name: body.id for body in self.bodies.values()}

    @cached_property
    def geom_name2id(self):
        geom_names = {geom.name for geom in self.geoms.values() if geom.name is not None}
        if len(geom_names) < len(self.geoms):
            print("WARN: Duplicate geom name or unnamed geom")
        return {geom.name: geom.id for geom in self.geoms.values() if geom.name is not None}

    @cached_property
    def joint_name2id(self):
        joint_names = {joint.name for joint in self.joints.values()}
        if len(joint_names) < len(self.joints):
            raise ValueError('Duplicate joint name')
        return {joint.name: joint.id for joint in self.joints.values()}

    @cached_property
    def site_name2id(self):
        site_names = {site.name for site in self.sites.values()}
        if len(site_names) < len(self.sites):
            raise ValueError('Duplicate site name')
        return {site.name: site.id for site in self.sites.values()}

    @cached_property
    def free_qpos_mask(self):
        independent = set()
        dependent = set()
        mask = np.ones(self.nq, dtype=bool)
        for eq in self.equalities:
            assert eq.joint1 not in independent and eq.joint1 not in dependent
            assert eq.joint2 not in dependent
            dependent.add(eq.joint1)
            independent.add(eq.joint2)
            pos1 = unit_slice(self.joint_slices[eq.joint1])
            mask[pos1] = False
        return mask

    def resolve_pose(self, qpos: np.ndarray, base_pose: Transform = None):
        body_poses: dict[int, Transform] = {}
        geom_poses: dict[int, Transform] = {}
        site_poses: dict[int, Transform] = {}

        cursor = 0
        def take(n):
            nonlocal cursor
            q = qpos[cursor:cursor+n]
            cursor += n
            return q
        # body pass
        for i, body in self.bodies.items():
            if cursor == len(qpos) and (body.nq > 0 or body.parent not in body_poses):
                # print(f"WARN: qpos is too short for bodies starting from {body.name}")
                continue
            if body.id == self.root:
                if body.is_free:
                    assert base_pose is None
                if base_pose is None:
                    parent_pose = Transform(np.zeros(3), np.array([1, 0, 0, 0]))
                else:
                    parent_pose = base_pose
            else:
                parent_pose = body_poses[body.parent]
            transforms = [parent_pose, body.initial_transform]
            for joint in body.joints:
                transforms.append(joint.to_transform(take(joint.nq)))
            pose = Transform.compose(*transforms)
            body_poses[i] = pose
        assert cursor == len(qpos)
        # geom pass
        for i, body in self.bodies.items():
            if i not in body_poses:
                continue
            for geom in body.geoms:
                geom_pose = Transform.compose(body_poses[i], geom.transform)
                geom_poses[geom.id] = geom_pose
            for site in body.sites:
                site_pose = Transform.compose(body_poses[i], site.transform)
                site_poses[site.id] = site_pose
        return body_poses, geom_poses, site_poses

    def resolve_sympose(self, base_pose: Transform = None):
        qpos = sp.ImmutableDenseNDimArray(sp.symbols(f'q:{self.nq}'))
        body_poses: dict[int, Transform] = {}
        geom_poses: dict[int, Transform] = {}
        site_poses: dict[int, Transform] = {}

        cursor = 0
        def take(n):
            nonlocal cursor
            q = qpos[cursor:cursor+n]
            cursor += n
            return q
        # body pass
        for i, body in self.bodies.items():
            if body.id == self.root:
                if len(body.joints) == 0 and base_pose is None:
                    parent_pose = Transform(sp.ImmutableDenseNDimArray([0, 0, 0]), sp.ImmutableDenseNDimArray([1, 0, 0, 0]))
                elif body.is_free:
                    assert base_pose is None
                    parent_pose = Transform(sp.ImmutableDenseNDimArray([0, 0, 0]), sp.ImmutableDenseNDimArray([1, 0, 0, 0]))
                else:
                    assert base_pose is not None
                    parent_pose = base_pose
            else:
                parent_pose = body_poses[body.parent]
            initial_transform = body.initial_transform
            initial_quat = []
            for element in initial_transform.quat:
                if np.isclose(element, 0.0):
                    initial_quat.append(sp.S.Zero)
                elif np.isclose(element, 1.0):
                    initial_quat.append(sp.S.One)
                elif np.isclose(element, 0.5):
                    initial_quat.append(sp.S.Half)
                elif np.isclose(element, -1.0):
                    initial_quat.append(-sp.S.One)
                elif np.isclose(element, -0.5):
                    initial_quat.append(-sp.S.Half)
                elif np.isclose(element, 0.5**0.5):
                    initial_quat.append(sp.sqrt(2)/2)
                elif np.isclose(element, -0.5**0.5):
                    initial_quat.append(-sp.sqrt(2)/2)
                else:
                    initial_quat.append(element)
            initial_transform = Transform(initial_transform.pos, sp.ImmutableDenseNDimArray(initial_quat))

            transforms = [parent_pose, initial_transform]
            for joint in body.joints:
                transforms.append(joint.to_transform(take(joint.nq)))
            pose = Transform.compose(*transforms)

            def encapulate(expr, prefix):
                args = sorted(expr.free_symbols, key=lambda x: int(x.name[1:]))
                return K.derive(f"{prefix}{i}", expr, *args)
            pos = sp.ImmutableDenseNDimArray([encapulate(e, p) for e, p in zip(pose.pos, ("X", "Y", "Z"))])
            quat = sp.ImmutableDenseNDimArray([encapulate(e, p) for e, p in zip(pose.quat, ("w", "x", "y", "z"))])
            body_poses[i] = Transform(pos, quat)
        assert cursor == len(qpos)
        # geom pass
        for i, body in self.bodies.items():
            for geom in body.geoms:
                geom_pose = Transform.compose(body_poses[i], geom.transform)
                geom_poses[geom.id] = geom_pose
                # def encapulate(expr, prefix):
                #     args = sorted(expr.free_symbols, key=lambda x: int(x.name[1:]))
                #     return K.derive(f"{prefix}_{i}@{j}", expr, *args)
                # pos = sp.ImmutableDenseNDimArray([encapulate(e, p) for e, p in zip(geom_pose.pos, ("X", "Y", "Z"))])
                # quat = sp.ImmutableDenseNDimArray([encapulate(e, p) for e, p in zip(geom_pose.quat, ("w", "x", "y", "z"))])
                # geom_poses[j] = Transform(pos, quat)
            for site in body.sites:
                site_pose = Transform.compose(body_poses[i], site.transform)
                site_poses[site.id] = site_pose
        return qpos, body_poses, geom_poses, site_poses

    def enforce_equality(self, qpos):
        independent = set()
        dependent = set()
        is_array = isinstance(qpos, sp.ImmutableDenseNDimArray)
        if is_array:
            qpos = qpos.tolist()
        for eq in self.equalities:
            assert eq.joint1 not in independent and eq.joint1 not in dependent
            assert eq.joint2 not in dependent
            dependent.add(eq.joint1)
            independent.add(eq.joint2)
            pos1 = unit_slice(self.joint_slices[eq.joint1])
            pos2 = unit_slice(self.joint_slices[eq.joint2])
            qpos[pos1] = eq.compute_joint1(qpos[pos2])
        if is_array:
            return sp.ImmutableDenseNDimArray(qpos)
        return qpos

    def expand_qpos(self, qpos):
        nq = self.nq
        nf = np.count_nonzero(self.free_qpos_mask)
        free_qpos_mask = self.free_qpos_mask
        assert nf == len(qpos)

        full_qpos = [None] * nq
        cursor = 0
        for i in range(nq):
            if free_qpos_mask[i]:
                full_qpos[i] = qpos[cursor]
                cursor += 1
        assert cursor == nf, (cursor, nf)
        qpos = self.enforce_equality(full_qpos)
        return qpos

    def visualize(self, ax: 'Axes3D', body_poses: dict[int, Transform], geom_poses: dict[int, Transform]):
        ax: Axes3D

        for i, body in self.bodies.items():
            is_root = body.id == self.root
            pose = body_poses[i]
            ax.scatter(*pose.pos, s=100, label=body.name, c='r')
            ax.text(*pose.pos, body.name)
            if not is_root:
                parent_pose = body_poses[body.parent]
                ax.plot(*np.array([parent_pose.pos, pose.pos]).T, c='g')
            # for geom in body.geoms:
            #     geom_pose = geom_poses[geom.id]
            #     ax.scatter(*geom_pose.pos, s=50, label=geom.name, c='b')
            #     ax.plot(*np.array([pose.pos, geom_pose.pos]).T, c='b')

        ax.set_aspect('equal')

    def visualize_3d(
        self,
        body_poses: dict[int, Transform],
        geom_poses: dict[int, Transform],
        site_poses: dict[int, Transform],
        *,
        world_axis: bool = True,
        body_axis: bool = False,
        geom_axis: bool = False,
    ):
        import trimesh
        scene = trimesh.Scene()
        if world_axis:
            world_axis = trimesh.creation.axis(origin_size=0.01)
            scene.add_geometry(world_axis, geom_name="axis/world")
        for i, body in self.bodies.items():
            if body_axis:
                body_pose = body_poses[i]
                axis = trimesh.creation.axis(origin_size=0.004, transform=body_pose.mat)
                scene.add_geometry(axis, geom_name=f"axis/body/{body.id}")
            for geom in body.geoms:
                if geom.id not in geom_poses:
                    continue
                geom_pose = geom_poses[geom.id]
                geom_mesh = geom.as_trimesh(transform=geom_pose.mat)
                scene.add_geometry(geom_mesh, geom_name=f"geom/{geom.id}")
                if geom_axis:
                    geom_axis = trimesh.creation.axis(origin_size=0.004, transform=geom_pose.mat)
                    scene.add_geometry(geom_axis, geom_name=f"axis/geom/{geom.id}")
            for site in body.sites:
                if site.id not in site_poses:
                    continue
                site_pose = site_poses[site.id]
                site_axis = trimesh.creation.axis(origin_size=0.004, transform=site_pose.mat)
                scene.add_geometry(site_axis, geom_name=f"axis/site/{site.id}")
        return scene

def build_hierarchy(
        model: mujoco.MjModel, root: int,
        visual_groups: tuple[int, ...] = (1, 2), collision_groups: tuple[int, ...] = (3, 4, 5)
    ) -> Hierarchy:
    assert 0 <= root < model.nbody
    # if model.ntendon > 0:
    #     print(f"WARN: Skipping {model.ntendon} tendons")
    bodies = {root: build_body(model, root, root=True, visual_groups=visual_groups, collision_groups=collision_groups)}
    for i in range(root+1, model.nbody):
        if model.body_parentid[i].item() not in bodies:
            # end of body hierarchy
            break
        body = build_body(model, i, root=False, visual_groups=visual_groups, collision_groups=collision_groups)
        bodies[i] = body
    joint_ids = {joint.id for body in bodies.values() for joint in body.joints}
    equalities = []
    for i in range(model.neq):
        eq = build_equality(model, i)
        if eq is not None:
            if eq.joint1 in joint_ids and (eq.joint2 is None or eq.joint2 in joint_ids):
                equalities.append(eq)
            elif eq.joint1 in joint_ids or eq.joint2 in joint_ids:
                print(f"Skipping invalid equality constraint {eq}")
            # Otherwise, it is a irrelevant equality constraint
    joint_ids = sorted(joint_ids)
    assert joint_ids == list(range(joint_ids[0], joint_ids[-1]+1))
    start_joint = joint_ids[0]
    end_joint = joint_ids[-1]
    start_q = model.jnt_qposadr[start_joint].item()
    end_q = model.jnt_qposadr[end_joint].item()
    end_q += {
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
        mujoco.mjtJoint.mjJNT_BALL: 4,
        mujoco.mjtJoint.mjJNT_FREE: 7,
    }[model.jnt_type[end_joint]]
    assert end_q - start_q == sum(body.nq for body in bodies.values())
    qs = slice(start_q, end_q)
    return Hierarchy(bodies, equalities, root, qs)
