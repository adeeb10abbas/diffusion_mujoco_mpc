import multiprocessing as mp
import mujoco
import numpy as np
import os
import pathlib
import torch
import gc

os.environ["MUJOCO_GL"] = "glfw"

from mujoco_mpc import agent as agent_lib
import os
def run_trajectory(traj_id, T, NUM_KNOT_POINTS, model_path):
    # model
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # data
    data = mujoco.MjData(model)

    # agent
    agent = agent_lib.Agent(task_id="Cartpole", model=model)

    # weights
    agent.set_cost_weights({"Velocity": 0.15})

    # parameters
    agent.set_task_parameter("Goal", -1.0)

    # Initialize arrays
    qpos = np.zeros((model.nq, T))
    qvel = np.zeros((model.nv, T))
    ctrl = np.zeros((model.nu, T - 1))
    policy_parameters = np.zeros((model.nu, NUM_KNOT_POINTS, T - 1))
    time = np.zeros(T)
    cost_total = np.zeros(T - 1)
    cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

    # reset data
    mujoco.mj_resetData(model, data)

    # cache initial state
    qpos[:, 0] = data.qpos
    qvel[:, 0] = data.qvel
    time[0] = data.time

    # Lists to save trajectory data
    positions = []
    vels = []
    spline_params = []
    u = []

    for t in range(T - 1):
        if t % 100 == 0:
            print(f"Trajectory {traj_id}, t = {t}")

        # set planner state
        agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )

        # run planner for num_steps
        num_steps = 1
        for _ in range(num_steps):
            agent.planner_step()

        # set ctrl from agent policy
        data.ctrl = agent.get_action()
        ctrl[:, t] = data.ctrl

        # set policy parameters
  #      policy_parameters[:, :, t] = agent.get_policy_parameters().reshape(model.nu, NUM_KNOT_POINTS)

        # Append data to lists
        positions.append(data.qpos.copy())
        vels.append(data.qvel.copy())
 #       spline_params.append(policy_parameters[:, :, t].copy())
        u.append(data.ctrl.copy())

        # get costs
        cost_total[t] = agent.get_total_cost()
        for i, c in enumerate(agent.get_cost_term_values().items()):
            cost_terms[i, t] = c[1]

        # step
        mujoco.mj_step(model, data)

        # cache
        qpos[:, t + 1] = data.qpos
        qvel[:, t + 1] = data.qvel
        time[t + 1] = data.time

    # Save trajectory data
    torch.save({
        "pos": positions,
        "vel": vels,
#        "spline_params": spline_params,
        "u": u
    }, f"trajectories/cartpole_traj_{traj_id}.pkl")

    # Reset agent and perform garbage collection
    agent.reset()
    agent.close()
    gc.collect(0)

# Main script execution
if __name__ == "__main__":
    # Model path
    model_path = pathlib.Path(__file__).parent.parent.parent / "../../build/mjpc/tasks/cartpole/task.xml"
    
    # Number of trajectories
    NUM_TRAJ = 4000
    
    # Rollout horizon
    T = 1500
    
    # Number of spline parameters
    NUM_KNOT_POINTS = 10
    
    # Create a process pool
    processes = []
    
    for traj_id in range(NUM_TRAJ):
        p = mp.Process(target=run_trajectory, args=(traj_id, T, NUM_KNOT_POINTS, model_path))
        processes.append(p)
        p.start()
        p.join()
