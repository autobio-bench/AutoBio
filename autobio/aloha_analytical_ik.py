import numpy as np
from grasp.quat import quatapply, quatinv, quatcompose

def close_to_zero(x: float, *, tol: float = 1e-8) -> bool:
    return -tol < x < tol

def site_pose_to_wrist_pos(site_pos: np.ndarray, site_quat: np.ndarray) -> np.ndarray:
    # Step 1: Convert site pose (left/gripper or gripper_site) to wrist position (left/wrist)
    #         leveraging the property of spherical wrist

    # For left/gripper
    local_axis = np.array([0.9998872305243953, 0.0, -0.015017530897414616])
    length = 0.1997665275665571
    # For gripper_site
    # local_axis = np.array([0.0, 0.0, 1.0])
    # length = 0.204744

    axis = quatapply(site_quat, local_axis)
    return site_pos - axis * length

def solve_base(wrist_pos: np.ndarray) -> np.ndarray | None:
    # Step 2: Solve base joint angles q0, q1, q2 given wrist_pos x, y, z

    # x = cos(q0) * (a * sin(q1) + b * cos(q1) + a * cos(q1 + q2))
    # y = sin(q0) * (a * sin(q1) + b * cos(q1) + a * cos(q1 + q2))
    # z = a * cos(q1) - b * sin(q1) - a * sin(q1 + q2) + c
    # where a = 0.3, b = 0.05955, c = 0.12705
    a, b, c = 0.3, 0.05955, 0.12705

    x, y, z = wrist_pos
    v = z - c

    def leftpad(arr: np.ndarray, v: float) -> np.ndarray:
        m, n = arr.shape
        res = np.empty((m, n + 1))
        res[:, 0] = v
        res[:, 1:] = arr
        return res

    if close_to_zero(x) and close_to_zero(y):
        # Special case: left/wrist almost upright
        # in this case, q0 is indeterminate
        q12 = solve_planar_2link(0.0, v)
        if q12 is None:
            return None
        # return leftpad(q12, np.nan)
        raise NotImplementedError
    else:
        # q0 and q0 + pi
        sln = []
        q0 = np.arctan2(y, x)
        u = np.sqrt(x ** 2 + y ** 2)
        q12 = solve_planar_2link(u, v)
        if q12 is not None:
            sln.append(leftpad(q12, q0))
        q12 = solve_planar_2link(-u, v)
        if q12 is not None:
            sln.append(leftpad(q12, q0 + np.pi))
        if len(sln) == 0:
            return None
        sln = np.concatenate(sln, axis=0)

        # validate
        # q0, q1, q2 = sln[:, 0], sln[:, 1], sln[:, 2]
        # print(np.cos(q0) * (a * np.sin(q1) + b * np.cos(q1) + a * np.cos(q1 + q2)), x)
        # print(np.sin(q0) * (a * np.sin(q1) + b * np.cos(q1) + a * np.cos(q1 + q2)), y)
        # print(a * np.cos(q1) - b * np.sin(q1) - a * np.sin(q1 + q2) + c, z)
        
        return sln

def solve_planar_2link(u: float, v: float) -> np.ndarray | None:
    # Step 2.1: Solve base joint angles q1, q2 given u, v
    a, b = 0.3, 0.05955

    # u = a * sin(q1) + b * cos(q1) + a * cos(q1 + q2)
    # v = a * cos(q1) - b * sin(q1) - a * sin(q1 + q2)
    # Rearrange to:
    # a * cos(q1 + q2) =  u - a * sin(q1) - b * cos(q1)
    # a * sin(q1 + q2) = -v + a * cos(q1) - b * sin(q1)
    # Simplify to:
    # (-2 * a * u + 2 * b * v) * sin(q1) + (-2 * a * v - 2 * b * u) * cos(q1) + b ** 2 + u ** 2 + v ** 2 = 0
    A = -2 * a * u + 2 * b * v
    B = -2 * a * v - 2 * b * u
    C = b ** 2 + u ** 2 + v ** 2
    q1 = solve_linear_trig(A, B, C)

    if q1 is None:
        return None

    # cos(q1 + q2) =  (u - a * sin(q1) - b * cos(q1)) / a
    # sin(q1 + q2) = -(v - a * cos(q1) + b * sin(q1)) / a
    cos_q12 =  (u - a * np.sin(q1) - b * np.cos(q1)) / a
    sin_q12 = -(v - a * np.cos(q1) + b * np.sin(q1)) / a
    q2 = np.arctan2(sin_q12, cos_q12) - q1

    # validate
    # print(a * np.sin(q1) + b * np.cos(q1) + a * np.cos(q1 + q2), u)
    # print(a * np.cos(q1) - b * np.sin(q1) - a * np.sin(q1 + q2), v)

    return np.stack((q1, q2), axis=-1)

def solve_linear_trig(A: float, B: float, C: float) -> np.ndarray | None:
    # Step 2.1.1: Solve linear trigonometric equation of the form:
    # A * sin(x) + B * cos(x) + C = 0
    assert not close_to_zero(A) or not close_to_zero(B)  # Should not happen
    
    # R * sin(x + theta) + C = 0
    R = np.sqrt(A ** 2 + B ** 2)
    theta = np.arctan2(B, A)

    # sin(x + theta) = t
    t = -C / R

    if np.abs(t) > 1:
        # No solution
        return
    
    # Otherwise, x = arcsin(t) - theta + k * 2 pi  OR  x = pi - arcsin(t) - theta + k * 2 pi
    inv = np.arcsin(t)

    # validate
    # print(A * np.sin(inv - theta) + B * np.cos(inv - theta) + C)
    # print(A * np.sin(np.pi - inv - theta) + B * np.cos(np.pi - inv - theta) + C)

    return np.array((inv - theta, np.pi - inv - theta))

def target_quat_in_wrist_frame(target_quat: np.ndarray, q_base: np.ndarray):
    # Step 3: Convert target_quat to wrist frame
    # wrist_quat:
    # w =  cos(q0/2) * cos(q1/2 + q2/2)
    # x = -sin(q0/2) * sin(q1/2 + q2/2)
    # y =  cos(q0/2) * sin(q1/2 + q2/2)
    # z =  sin(q0/2) * cos(q1/2 + q2/2)
    
    # Step 3.1: Wrist FK
    q0, q1, q2 = q_base[..., 0], q_base[..., 1], q_base[..., 2]
    c0 = np.cos(q0 / 2)
    s0 = np.sin(q0 / 2)
    c12 = np.cos(q1 / 2 + q2 / 2)
    s12 = np.sin(q1 / 2 + q2 / 2)
    wrist_quat = np.stack((c0 * c12, -s0 * s12, c0 * s12, s0 * c12), axis=-1)

    # # Inverse left/gripper_base body to left/gripper_link body used by solve_wrist
    # eef_body_quat_inv = quatinv(np.array([0.5, -0.5, 0.5, -0.5]))

    # Step 3.2: Convert target_quat to wrist frame
    target_quat_wrist_local = [
        # quatcompose(quatinv(quat), quatcompose(target_quat, eef_body_quat_inv))
        quatcompose(quatinv(quat), target_quat)
        for quat in wrist_quat
    ]
    
    return np.array(target_quat_wrist_local)

def solve_wrist(target_quat: np.ndarray):
    # Step 4: Solve wrist joint angles q3, q4, q5 given target_quat in wrist frame
    # w = cos(q4/2) * cos(q3/2 + q5/2)
    # x = cos(q4/2) * sin(q3/2 + q5/2)
    # y = sin(q4/2) * cos(q3/2 - q5/2)
    # z = sin(q4/2) * sin(q3/2 - q5/2)

    w, x, y, z = target_quat

    # Solve q4 first, assume range of q4 is smaller than [-pi, pi]
    c4_sqr = w**2 + x**2  # cos^2(q4/2)
    p35 = np.arctan2(x, w)  # q3/2 + q5/2
    if close_to_zero(1 - c4_sqr):
        # Special case: when cos^2(q4/2) is near 1, q3 and q5 are indeterminate (gimbal lock)
        q4 = 0.0
        q3 = 0.0
        q5 = p35 * 2
        return np.array(((q3, q4, q5),))
    
    q4 = np.arccos(np.clip(2 * c4_sqr - 1, -1.0, 1.0))
    m35 = np.arctan2(z, y)  # q3/2 - q5/2 
    sln = np.array((
        (p35 + m35, q4, p35 - m35),
        (p35 + m35 + np.pi, -q4, p35 - m35 - np.pi)
    ))

    # sln[:, 1] -= np.pi / 2
    # sln[:, 2] -= np.pi / 2

    # Validate
    # q3, q4, q5 = sln[:, 0], sln[:, 1], sln[:, 2]
    # print(np.cos(q4 / 2) * np.cos(q3 / 2 + q5 / 2), w)
    # print(np.cos(q4 / 2) * np.sin(q3 / 2 + q5 / 2), x)
    # print(np.sin(q4 / 2) * np.cos(q3 / 2 - q5 / 2), y)
    # print(np.sin(q4 / 2) * np.sin(q3 / 2 - q5 / 2), z)

    return sln

def prune_by_bounds(sln: np.ndarray):
    bounds = np.array([
        [-3.14158,  3.14158],
        [-1.85005,  1.25664],
        [-1.76278,  1.6057 ],
        [-3.14158,  3.14158],
        [-1.8675 ,  2.23402],
        [-6.28   ,  6.28   ],
    ])
    sln = (sln + np.pi) % (2 * np.pi) - np.pi
    mask = np.all((sln >= bounds[:, 0]) & (sln <= bounds[:, 1]), axis=1)
    return sln[mask]

def aloha_analytical_ik(target_pos: np.ndarray, target_quat: np.ndarray):
    wrist_pos = site_pose_to_wrist_pos(target_pos, target_quat)
    q_base = solve_base(wrist_pos)
    if q_base is None:
        return None
    target_quat_wrist_local = target_quat_in_wrist_frame(target_quat, q_base)

    def leftpad(arr: np.ndarray, v: np.ndarray) -> np.ndarray:
        m, n = arr.shape
        res = np.empty((m, n + len(v)))
        res[:, 0:len(v)] = v
        res[:, len(v):] = arr
        return res
    
    sln = []
    for q, quat in zip(q_base, target_quat_wrist_local):
        q_wrist = solve_wrist(quat)
        sln.append(leftpad(q_wrist, q))
    sln = np.concatenate(sln, axis=0)

    sln = prune_by_bounds(sln)

    return sln
