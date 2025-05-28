from functools import wraps
import math

import numpy as np
import sympy as sp
import jax, jax.numpy as jnp

def wrapmath(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        mode = 'numpy'
        def set_mode(new_mode):
            nonlocal mode
            if new_mode == mode:
                return
            elif mode == 'numpy':
                mode = new_mode
            else:
                raise ValueError(f'cannot mix {mode} with {new_mode}')

        def one_arg(arg):
            nonlocal mode
            if isinstance(arg, np.ndarray):
                if arg.dtype == np.object_:
                    set_mode('sympy')
                return arg.tolist()
            elif isinstance(arg, np.number):
                return arg.item()
            elif isinstance(arg, sp.ImmutableDenseNDimArray):
                set_mode('sympy')
                return arg.tolist()
            elif isinstance(arg, sp.Symbol):
                set_mode('sympy')
                return arg
            elif isinstance(arg, jax.Array):
                set_mode('jax')
                return arg
            else:
                raise TypeError(f'unsupported type {type(arg)}')
        args = list(map(one_arg, args))
        ret = func(mode, *args, **kwargs)
        if mode == 'sympy':
            return sp.ImmutableDenseNDimArray(ret)
        elif mode == 'jax':
            return jnp.array(ret)
        else:
            return np.array(ret)
    return wrapper

@wrapmath
def quat2rot(_, q, /):
    w, x, y, z = q
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    xy, zw, xz, yw, yz, xw = x*y, z*w, x*z, y*w, y*z, x*w
    return [[w2+x2-y2-z2, 2*(xy-zw), 2*(xz+yw)],
            [2*(xy+zw), w2-x2+y2-z2, 2*(yz-xw)],
            [2*(xz-yw), 2*(yz+xw), w2-x2-y2+z2]]

@wrapmath
def quatcompose(_, q1, q2, /):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return [w, x, y, z]

@wrapmath
def quatapply(_, q, v, /):
    w, x, y, z = q
    x0, y0, z0 = v
    # (xyz cross xyz0) * 2
    x1 = (y*z0 - z*y0) * 2
    y1 = (z*x0 - x*z0) * 2
    z1 = (x*y0 - y*x0) * 2
    # xyz cross xyz1
    x2 = y*z1 - z*y1
    y2 = z*x1 - x*z1
    z2 = x*y1 - y*x1

    x1 = x0 + w*x1 + x2
    y1 = y0 + w*y1 + y2
    z1 = z0 + w*z1 + z2
    return [x1, y1, z1]

@wrapmath
def quatinv(_, q, /):
    w, x, y, z = q
    return [w, -x, -y, -z]

@wrapmath
def axisangle2quat(mode, axis, angle, /):
    x, y, z = axis
    half = angle / 2
    if mode == 'sympy':
        c = sp.cos(half)
        s = sp.sin(half)
    elif mode == 'jax':
        c = jnp.cos(half)
        s = jnp.sin(half)
    else:
        c = math.cos(half)
        s = math.sin(half)
    return [c, s*x, s*y, s*z]

if __name__ == '__main__':
    from scipy.spatial.transform import Rotation as R
    q1 = np.array([np.sqrt(1/11), np.sqrt(2/11), np.sqrt(3/11), np.sqrt(5/11)])
    q2 = np.array([np.sqrt(1/13), np.sqrt(3/13), np.sqrt(4/13), np.sqrt(5/13)])
    R1 = R(q1, scalar_first=True, normalize=False)
    R2 = R(q2, scalar_first=True, normalize=False)
    v = np.array([1, 2, 3])
    assert np.allclose(quat2rot(q1), R1.as_matrix())
    assert np.allclose(quat2rot(q2), R2.as_matrix())
    assert np.allclose(quatcompose(q1, q2), (R1 * R2).as_quat(scalar_first=True))
    assert np.allclose(quatapply(q1, v), R1.apply(v))
