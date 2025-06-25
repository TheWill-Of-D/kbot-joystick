"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import HumanoidWalkingTask, Model

# The model expects a command vector with 7 elements.
NUM_COMMANDS_MODEL = 7


def rotate_quat_by_quat(quat_to_rotate: Array, rotating_quat: Array, inverse: bool = False, eps: float = 1e-6) -> Array:
    """Rotates one quaternion by another quaternion through quaternion multiplication."""
    quat_to_rotate = quat_to_rotate / (jnp.linalg.norm(quat_to_rotate, axis=-1, keepdims=True) + eps)
    rotating_quat = rotating_quat / (jnp.linalg.norm(rotating_quat, axis=-1, keepdims=True) + eps)

    if inverse:
        rotating_quat = rotating_quat.at[..., 1:].multiply(-1)

    w1, x1, y1, z1 = jnp.split(rotating_quat, 4, axis=-1)
    w2, x2, y2, z2 = jnp.split(quat_to_rotate, 4, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    result = jnp.concatenate([w, x, y, z], axis=-1)

    return result / (jnp.linalg.norm(result, axis=-1, keepdims=True) + eps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: HumanoidWalkingTask = HumanoidWalkingTask.load_task(ckpt_path)
    model: Model = task.load_ckpt(ckpt_path, part="model")[0]

    mujoco_model = task.get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]

    # The model's carry state is composite: it contains the RNN's carry and the smoothed quaternion.
    rnn_carry_shape = (task.config.depth, task.config.hidden_size)
    imu_obs_carry_shape = (4,)  # for the smoothed quaternion
    total_carry_size = rnn_carry_shape[0] * rnn_carry_shape[1] + imu_obs_carry_shape[0]

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=NUM_COMMANDS_MODEL,
        carry_size=(total_carry_size,),
    )

    @jax.jit
    def init_fn() -> Array:
        """Initializes the flat carry state."""
        return jnp.zeros((total_carry_size,))

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        quaternion: Array,
        command: Array,
        gyroscope: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        """Performs one step of the model inference."""
        # --- Unpack the carry state ---
        rnn_carry_flat = carry[:rnn_carry_shape[0] * rnn_carry_shape[1]]
        rnn_carry = rnn_carry_flat.reshape(rnn_carry_shape)
        prev_smoothed_quat = carry[rnn_carry_shape[0] * rnn_carry_shape[1]:]

        # --- Replicate ImuOrientationObservation logic ---
        # Note: kinfer-sim's provider zeros out command[3], so heading_yaw_cmd will be 0.
        # This is a deviation from how the model was likely trained.
        # For a more robust solution, kinfer-sim/provider.py should be updated.
        heading_yaw_cmd = command[3]
        heading_yaw_cmd_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, heading_yaw_cmd]))
        backspun_framequat = rotate_quat_by_quat(quaternion, heading_yaw_cmd_quat, inverse=True)
        backspun_framequat = jnp.where(backspun_framequat[0] < 0, -backspun_framequat, backspun_framequat)

        # --- Implement the EMA filter (smoothing) ---
        lag = 0.05
        smoothed_quat = prev_smoothed_quat * lag + backspun_framequat * (1 - lag)

        # --- Construct the observation vector for the model ---
        cmd_vel = command[:2]
        cmd_yaw_rate = command[2:3]
        cmd_body_height = command[4:5]
        cmd_body_orientation = command[5:7]

        obs = jnp.concatenate(
            [
                joint_angles,
                joint_angular_velocities,
                smoothed_quat,
                cmd_vel,
                cmd_yaw_rate,
                jnp.zeros_like(command[3:4]),  # This was masked during training
                cmd_body_height,
                cmd_body_orientation,
                gyroscope,
            ],
            axis=-1,
        )
        dist, next_rnn_carry = model.actor.forward(obs, rnn_carry)

        # --- Pack the next carry state ---
        next_carry_flat = jnp.concatenate([next_rnn_carry.flatten(), smoothed_quat])

        return dist.mode(), next_carry_flat

    # Export the functions for the kinfer model
    init_onnx = export_fn(model=init_fn, metadata=metadata)
    step_onnx = export_fn(model=step_fn, metadata=metadata)

    kinfer_model = pack(
        init_fn=init_onnx,
        step_fn=step_onnx,
        metadata=metadata,
    )

    # Saves the resulting model
    (output_path := Path(args.output_path)).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_model)
    
    print(f"Successfully converted model to {args.output_path}")


if __name__ == "__main__":
    main()