# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import mediapy as media
import mujoco
import numpy as np
import pathlib

import predictive_sampling
# %%
# path to hand task

model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../build/mjpc/tasks/cartpole/task.xml"
)
# create simulation model + data
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)


# %%
# reward

## TODO: (@ali-bdai) - write your reward function 
## everything is a cost
## 
"""
void Cartpole::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  // ---------- Vertical ----------
  residual[0] = std::cos(data->qpos[1]) - 1;

  // ---------- Centered ----------
  residual[1] = data->qpos[0] - parameters_[0];

  // ---------- Velocity ----------
  residual[2] = data->qvel[1];

  // ---------- Control ----------
  residual[3] = data->ctrl[0];
}

6 - smoothabsloss ("mjpc/norm.h") sqrt(x^2+p^2) - p (where p = 0.1)
  - vertical 
  - centered
0 - quadratic (0.5x^2)
  - velocity
  - control

- reimplement norms from cpp (norm.cpp)
- 
    <user name="Vertical" dim="1" user="6 10.0 0 100.0 0.01"/>
    <user name="Centered" dim="1" user="6 10.0 0 100.0 0.1"/>
    <user name="Velocity" dim="1" user="0 0.1 0.0 1.0"/>
    <user name="Control" dim="1" user="0 0.1 0.0 1.0"/>

These are the params -     <numeric name="residual_Goal" data="0.0 -1.5 1.5" />
so, params[0] is just 0

what each of them stand for is in the docs/

<sensor>
    <user
        name="[term_name]"
        dim="[residual_dimension]"
        user="
            [norm_type]
            [weight]
            [weight_lower_bound]
            [weight_upper_bound]
            [norm_parameters...]"
    />

"""
import math

def smooth_abs_loss_norm(x, p):
  return math.sqrt(x**2 + p**2) - p

def quadratic_norm(x):
  return 0.5*x**2

def reward(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  p = 0.01
  position = data.sensor("position").data
  velocity = data.sensor("velocity").data
  control = data.ctrl

  r_0 = smooth_abs_loss_norm(math.cos(position[0]) - 1, p = 0.01) # vertical 
  r_1 = smooth_abs_loss_norm(position[0], p = 0.1) # centered
  r_2 = quadratic_norm(velocity)# velocity
  r_3 = quadratic_norm(control)# control

  return 10*r_0 + 10*r_1 + 0.1*r_2 + 0.1*r_3

# def reward(model: mujoco.MjModel, data: mujoco.MjData) -> float:
#   # cube position - palm position (L22 norm)
#   pos_error = (
#       data.sensor("cube_position").data - data.sensor("palm_position").data
#   )
#   p = 0.02
#   q = 2.0
#   c = np.dot(pos_error, pos_error)
#   a = c ** (0.5 * q) + p**q
#   s = a ** (1 / q)
#   r0 = -(s - p)

#   # cube orientation - goal orientation
#   goal_orientation = data.sensor("cube_goal_orientation").data
#   cube_orientation = data.sensor("cube_orientation").data
#   subquat = np.zeros(3)
#   mujoco.mju_subQuat(subquat, goal_orientation, cube_orientation)
#   r1 = -0.5 * np.dot(subquat, subquat)

#   # cube linear velocity
#   linvel = data.sensor("cube_linear_velocity").data
#   r2 = -0.5 * np.dot(linvel, linvel)

#   # actuator
#   effort = data.actuator_force
#   r3 = -0.5 * np.dot(effort, effort)

#   # grasp
#   graspdiff = data.qpos[7:] - model.key_qpos[0][7:]
#   r4 = -0.5 * np.dot(graspdiff, graspdiff)

#   # joint velocity
#   jntvel = data.qvel[6:]
#   r5 = -0.5 * np.dot(jntvel, jntvel)

#   return 20.0 * r0 + 5.0 * r1 + 10.0 * r2 + 0.1 * r3 + 2.5 * r4 + 1.0e-4 * r5


# %%
# planner
horizon = 0.25
splinestep = 0.05
planstep = 0.01
nimprove = 10
nsample = 10
noise_scale = 0.1
interp = "zero"
planner = predictive_sampling.Planner(
    model,
    reward,
    horizon,
    splinestep,
    planstep,
    nsample,
    noise_scale,
    nimprove,
    interp=interp,
)
# %%
# simulate
mujoco.mj_resetData(model, data)
steps = 500

# history
qpos = [data.qpos]
qvel = [data.qvel]
act = [data.act]
ctrl = []
rewards = []

# frames
frames = []
FPS = 1.0 / model.opt.timestep

# verbose
VERBOSE = False

u_init = None
for _ in range(steps):
  ## predictive sampling

  # improve policy
  planner.improve_policy(
      data.qpos, data.qvel, data.act, data.time, data.mocap_pos, data.mocap_quat
  )

  # get action from policy
  data.ctrl = planner.action_from_policy(data.time)

  u_init = planner._parameters
  # data.ctrl = np.random.normal(scale=0.1, size=model.nu)

  # reward
  rewards.append(reward(model, data))

  if VERBOSE:
    print("time  : ", data.time)
    print(" qpos  : ", data.qpos)
    print(" qvel  : ", data.qvel)
    print(" act   : ", data.act)
    print(" action: ", data.ctrl)
    print(" reward: ", rewards[-1])

  # step
  mujoco.mj_step(model, data)

  # history
  qpos.append(data.qpos)
  qvel.append(data.qvel)
  act.append(data.act)
  ctrl.append(ctrl)

  # render and save frames
  renderer.update_scene(data)
  pixels = renderer.render()
  frames.append(pixels)
# %%
media.show_video(frames, fps=FPS)
