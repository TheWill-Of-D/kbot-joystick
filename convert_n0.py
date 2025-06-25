"""
Converts a checkpoint to a deployable model.
This version includes the EMA filter for the IMU observation to perfectly match training.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import HumanoidWalkingTask, Model

# --- Configuration ---
NUM_KINFER_COMMANDS = 6
CONTROL_DT = 0.02
# The EMA filter lag. In training, this was a random value between 0.0 and 0.1.
# We'll use a fixed midpoint value for deployment.
EMA_LAG = 0.05

# --- Quaternion Math (Scatter-safe for conversion) ---

def euler_to_quat(euler_3: Array) -> Array:
    """Converts roll, pitch, yaw angles to a quaternion (w, x, y, z)."""
    roll, pitch, yaw = jnp.split(euler_3, 3, axis=-1)
    cr, sr = jnp.cos(roll * 0.5), jnp.sin(roll * 0.5)
    cp, sp = jnp.cos(pitch * 0.5), jnp.sin(pitch * 0.5)
    cy, sy = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = jnp.concatenate([w, x, y, z], axis=-1)
    return quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)

def rotate_quat_by_quat(quat_to_rotate: Array, rotating_quat: Array, inverse: bool = False, eps: float = 1e-6) -> Array:
    """Rotates one quaternion by another quaternion."""
    quat_to_rotate = quat_to_rotate / (jnp.linalg.norm(quat_to_rotate, axis=-1, keepdims=True) + eps)
    rotating_quat = rotating_quat / (jnp.linalg.norm(rotating_quat, axis=-1, keepdims=True) + eps)
    
    if inverse:
        w_inv, x_inv, y_inv, z_inv = jnp.split(rotating_quat, 4, axis=-1)
        x_inv, y_inv, z_inv = -x_inv, -y_inv, -z_inv
        rotating_quat = jnp.concatenate([w_inv, x_inv, y_inv, z_inv], axis=-1)

    w1, x1, y1, z1 = jnp.split(rotating_quat, 4, axis=-1)
    w2, x2, y2, z2 = jnp.split(quat_to_rotate, 4, axis=-1)
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    result = jnp.concatenate([w, x, y, z], axis=-1)
    
    return result / (jnp.linalg.norm(result, axis=-1, keepdims=True) + eps)

# --- Main Conversion Logic ---

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

    # --- Define shapes for our comprehensive carry state ---
    gru_carry_shape = (task.config.depth, task.config.hidden_size)
    prev_cmd_shape = (NUM_KINFER_COMMANDS,)
    ema_quat_shape = (4,) # For the EMA filter state
    
    flat_gru_carry_size = gru_carry_shape[0] * gru_carry_shape[1]
    flat_carry_size = flat_gru_carry_size + prev_cmd_shape[0] + ema_quat_shape[0]

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=NUM_KINFER_COMMANDS,
        carry_size=(flat_carry_size,),
    )

    @jax.jit
    def init_fn() -> Array:
        """Initializes the flat carry state."""
        initial_gru_carry = jnp.zeros(gru_carry_shape)
        initial_prev_cmd = jnp.zeros(prev_cmd_shape)
        
        # --- THIS IS THE FIX ---
        # Initialize the EMA state to the identity quaternion [1, 0, 0, 0],
        # representing a stable "no error" starting state.
        initial_ema_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        # --- END OF FIX ---

        return jnp.concatenate([
            initial_gru_carry.flatten(),
            initial_prev_cmd,
            initial_ema_quat
        ])

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        quaternion: Array,
        command: Array,
        gyroscope: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        """The main step function for the deployed model."""
        # --- Unpack the carry state ---
        gru_carry_flat = carry[:flat_gru_carry_size]
        prev_command = carry[flat_gru_carry_size : flat_gru_carry_size + prev_cmd_shape[0]]
        prev_ema_quat = carry[flat_gru_carry_size + prev_cmd_shape[0]:]
        
        gru_carry = gru_carry_flat.reshape(gru_carry_shape)

        # --- Calculate back-spun quaternion (as before) ---
        yaw_absolute_cmd = command[2]
        heading_yaw_cmd_quat = euler_to_quat(jnp.array([0.0, 0.0, yaw_absolute_cmd]))
        backspun_imu_quat_raw = rotate_quat_by_quat(quaternion, heading_yaw_cmd_quat, inverse=True)
        backspun_imu_quat_raw = jnp.where(backspun_imu_quat_raw[0] < 0, -backspun_imu_quat_raw, backspun_imu_quat_raw)

        # --- NEW: Apply the EMA smoothing filter ---
        smoothed_imu_quat = prev_ema_quat * EMA_LAG + backspun_imu_quat_raw * (1 - EMA_LAG)

        # --- Construct command vector (as before) ---
        prev_yaw_absolute_cmd = prev_command[2]
        wz = (yaw_absolute_cmd - prev_yaw_absolute_cmd) / CONTROL_DT
        
        command_for_actor = jnp.concatenate([
            command[:2],
            jnp.array([wz]),
            jnp.array([0.0]),
            command[3:]
        ])

        # --- Assemble the final, correct observation vector ---
        obs_n = jnp.concatenate([
            joint_angles,
            joint_angular_velocities,
            smoothed_imu_quat, # Use the smoothed value!
            command_for_actor,
            gyroscope,
        ])

        # --- Run the actor model ---
        dist, next_gru_carry = model.actor.forward(obs_n, gru_carry)

        # --- Pack the new carry state for the next step ---
        next_carry_flat = jnp.concatenate([
            next_gru_carry.flatten(),
            command,
            smoothed_imu_quat
        ])

        return dist.mode(), next_carry_flat

    # Export and package the model
    init_onnx = export_fn(model=init_fn, metadata=metadata)
    step_onnx = export_fn(model=step_fn, metadata=metadata)

    kinfer_model = pack(
        init_fn=init_onnx,
        step_fn=step_onnx,
        metadata=metadata,
    )

    # Save the final .kinfer file
    (output_path := Path(args.output_path)).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_model)
    print(f"Successfully converted model to {output_path}")


if __name__ == "__main__":
    main()