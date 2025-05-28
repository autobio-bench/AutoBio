from pathlib import Path

import jax, jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")
import orbax.checkpoint as ocp

def load_params(path: Path, dtype=jnp.bfloat16):
    path = Path(path).resolve()
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(path)
        item = {"params": metadata["params"]}
        sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
        restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(dtype=dtype, sharding=sharding), item)
        params = ckptr.restore(path, ocp.args.PyTreeRestore(item=item, restore_args=restore_args))
    return params

def save_params(path: Path, params):
    path = Path(path).resolve()
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(path, params)

def shrink_checkpoint(path: Path, validate: bool = False):
    param_path = path / "params"
    if not param_path.exists():
        raise FileNotFoundError(f"Checkpoint params not found at {param_path}")
    backup_path = param_path.with_suffix(".bak")
    if backup_path.exists():
        raise FileExistsError(f"Backup params already exists at {backup_path}")
    
    params = load_params(param_path)
    param_path.rename(backup_path)
    save_params(param_path, params)

    if validate:
        new_params = load_params(param_path)
        assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), params, new_params)), "Params do not match after shrinking"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shrink a checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to the checkpoint directory")
    parser.add_argument("--validate", action="store_true", help="Validate the checkpoint after shrinking")
    args = parser.parse_args()

    shrink_checkpoint(args.checkpoint)
