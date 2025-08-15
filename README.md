This project were termporary stopped due to problems with getting the wheel torque data from the simulator. 
The neural network is not implemented

# Reinforcement Learning implementation for Husky robot with ROS and Gazebo

Reinforcement Learning (RL) implementation tailored for the Husky mobile robot, integrated with the Robot Operating System (ROS) and the Gazebo robotics simulator. The project focuses on developing and evaluating intelligent control strategies for the Husky robot in a simulated environment.

## Project Overview

By leveraging ROS for inter-process communication and Gazebo for realistic physics simulation, the project provides a robust platform for developing and testing autonomous robot behaviors. The core idea is to enable the robot to learn optimal policies through trial and error, guided by a reward system.

## Key Components and Functionality

### 1. Agent Implementation (`agent/` directory)

- **RL Algorithms:** Implementations of algorithms such as Q-learning, Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), or other suitable RL algorithms.
- **Policy Networks:** Neural networks that map observations from the robot's sensors to actions the robot can take.
- **Training Loop:** The main script that manages the interaction between the agent and the environment, including experience collection, model updates, and evaluation.

### 2. Environment Definition (`env/` directory)

- **Observation Space:** Defining what the robot perceives from its sensors (e.g., laser scan data, odometry, camera images).
- **Action Space:** Defining the set of actions the robot can perform (e.g., linear and angular velocities).
- **Reward Function:** Crafting a reward signal that guides the agent towards desired behaviors (e.g., reaching a target, avoiding obstacles, maintaining a certain speed).
- **Reset and Step Functions:** Standard Gym interface functions for resetting the environment to an initial state and taking a step based on an action.

### 3. ROS Integration (`ros/` directory or integrated within other components)

- **ROS Nodes:** Separate processes for different functionalities (e.g., sensor data processing, motor control, RL agent).
- **ROS Topics:** Communication channels for publishing sensor data and subscribing to control commands.
- **ROS Services:** For requesting specific actions or information from other ROS nodes.

### 4. Gazebo Simulation (`gazebo/` or integrated with environment)

- **Robot Model:** The URDF (Unified Robot Description Format) or XACRO (XML Macro) model of the Husky robot, defining its physical properties and sensors.
- **World Files:** XML files that describe the simulated environment, including obstacles, terrains, and other objects.
- **Physics Engine:** Gazebo's built-in physics engine simulates realistic robot dynamics and sensor readings.

### 5. Utility Scripts (`utils/` directory)

- **Data Logging:** For recording training progress, rewards, and other metrics.
- **Visualization Tools:** For visualizing the robot's behavior or sensor data.
- **Configuration Files:** Parameters for the RL agent, environment, and ROS nodes.

## Technical Stack

- **Python:** The primary programming language for RL algorithms and ROS scripting.
- **ROS (Robot Operating System):** For inter-process communication and robotics framework.
- **Gazebo:** For 3D robotics simulation.
- **OpenAI Gym:** For defining the reinforcement learning environment.
- **TensorFlow/PyTorch (Implied):** For building and training neural networks for the RL agent.
- **NumPy:** For numerical operations.

## Usage

To run this project, you would need a ROS and Gazebo installation. The workflow would involve:

1.  **Launching Gazebo:** Starting the Gazebo simulator with the Husky robot and the desired environment.
2.  **Running ROS Nodes:** Launching the necessary ROS nodes for sensor data, motor control, and the RL agent.
3.  **Training the Agent:** Executing the training script to allow the RL agent to learn from interactions with the simulated environment.
4.  **Evaluation:** Testing the trained agent's performance in various scenarios.

## Potential Applications

- **Autonomous Navigation:** Training robots to navigate complex environments.
- **Robotic Manipulation:** Developing agents for tasks involving object interaction.
- **Swarm Robotics:** Extending to multiple robots for collaborative tasks.
- **Human-Robot Interaction:** Building intelligent behaviors for robots interacting with humans.
- **Industrial Automation:** Applying RL to optimize processes in manufacturing and logistics.

## Getting Started

Call "export HUSKY_OUSTER_ENABLED=1" in the terminal before launch.

Look on some examples here:
https://ai-mrkogao.github.io/reinforcement%20learning/ROSRL/

and their github:
https://github.com/erlerobot/gym-gazebo/tree/master

Gym library documentation:
https://www.gymlibrary.dev/

ROS documentation:
http://wiki.ros.org/ROS/Tutorials
http://wiki.ros.org/reinforcement_learning/Tutorials/Reinforcement%20Learning%20Tutorial

