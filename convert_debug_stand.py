"""
A debug conversion script that creates a "dummy" policy.
This policy ignores all sensory input and simply outputs the default
joint positions for standing still.
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

# We import ZEROS to get the default joint angles
from train import ZEROS, HumanoidWalkingTask

def main() -> None:
    parser = argparse.ArgumentParser()
    # We still need the checkpoint path to load task metadata like joint names
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: HumanoidWalkingTask = HumanoidWalkingTask.load_task(ckpt_path)
    mujoco_model = task.get_mujoco_model()
    joint_names = task.get_joint_names(mujoco_model)

    # The carry state can be empty since this model is stateless
    carry_shape = (1,)

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=0, # This model takes no commands
        carry_size=carry_shape,
    )

    # A stateless init function that returns a dummy carry
    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    # The dummy step function. It IGNORES all inputs except the carry.
    @jax.jit
    def step_fn(carry: Array) -> tuple[Array, Array]:
        # Get the default "zero" positions for each joint
        default_positions = jnp.array([p for _, p, _ in ZEROS])

        # The output is always the default standing pose.
        # The carry is passed through unchanged.
        return default_positions, carry

    # Export and package the dummy model
    init_onnx = export_fn(model=init_fn, metadata=metadata)
    step_onnx = export_fn(model=step_fn, metadata=metadata)
    kinfer_model = pack(init_fn=init_onnx, step_fn=step_onnx, metadata=metadata)

    (output_path := Path(args.output_path)).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_model)
    print(f"Successfully converted DUMMY STAND model to {output_path}")

if __name__ == "__main__":
    main()