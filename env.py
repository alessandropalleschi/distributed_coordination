# env.py
import pybullet as p
import pybullet_data
import gym
from gym.spaces import Box, Dict
import torch
import time
import numpy as np
from collections import deque

class RunningStats:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.variance = np.zeros(shape)
        self.count = 0

    def update(self, x):
        batch_mean = np.mean(x, axis=0, keepdims=True)
        batch_variance = np.var(x, axis=0, keepdims=True)
        batch_count = x.shape[0]

        new_count = self.count + batch_count
        delta = batch_mean - self.mean

        self.mean = self.mean + delta * batch_count / new_count
        if new_count == 1:
            self.variance = batch_variance
        else:
            self.variance = (
                self.variance * (self.count - 1) + batch_variance * (batch_count - 1) +
                delta**2 * self.count * batch_count / new_count
            ) / (new_count - 1)

        self.count = new_count

    def get_stats(self):
        return self.mean, np.sqrt(self.variance)

class MultiRobotEnv(gym.Env):
    def __init__(self, num_robots=2, num_obstacles=5):
        super(MultiRobotEnv, self).__init__()

        self.max_history = 3
        self.wheel_distance = 0.23
        self.simulation_step = 1/240
        self.sampling_time = 0.1
        # Initialize PyBullet
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.num_robots = num_robots
        self.num_obstacles = num_obstacles
        self.current_pos = []
        self.prev_pos = []
        self.robot_ids = []
        self.obstacle_ids = []
        self.goal_positions = []
        self.radius = 0.2
        self.wheel_radius = 0.035
        self.range_measurements_history = {}
        # Define action and observation space
        self.action_space = Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float64)
        self.observation_space = Box(0.0, np.inf, shape=(4,), dtype=np.float64)
        self.min_distance = np.zeros(num_robots)
        self.sphere_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0, 0, 1]
        )
        self.wgoal = 2.5
        self.wpenalty = -0.1
        self.rarrival = 15
        self.rcoll = -15
        self.dist_tresh = 0.15
        self.commanded_vel = []
        self.iter = 0
        self.max_duration = 250
        self.target_link_id = 0

    def compute_running_stats(self, observations):
        # Compute running mean and standard deviation
        if not hasattr(self, 'running_stats'):
            self.running_stats = [RunningStats((1, 1, 512)) for _ in range(observations.shape[1])]

        for channel, stats in enumerate(self.running_stats):
            channel_data = observations[:, channel:channel+1, :]
            stats.update(channel_data)
        
    def normalize(self, observations):
        normalized_observations = np.zeros_like(observations)

        for channel, stats in enumerate(self.running_stats):
            channel_data = observations[:, channel:channel+1, :]
            mean, std = stats.get_stats()

            # Normalize along the feature dimension
            normalized_channel_data = (channel_data - mean) / std
            normalized_observations[:, channel:channel+1, :] = normalized_channel_data

        return normalized_observations
                
    def reset(self):
        # Reset the simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        # Re-initialize robots
        self.robot_ids = []
        self.prev_pos = []
        self.obstacle_ids = []
        self.goal_positions = []

        for _ in range(self.num_robots):
            pos = self.get_random_robot_position()
            ori = p.getQuaternionFromEuler((0,0,np.random.uniform(-np.pi,np.pi)))
            robot_id = p.loadURDF("turtlebot.urdf", pos, ori)
            # Specify the name of the link you want to identify
            target_link_name = "cliff_sensor_front_joint"
            num_joints = p.getNumJoints(robot_id)
            # Loop through the joints to find the link ID
            for i in range(num_joints):
                joint_info = p.getJointInfo(robot_id, i)
                if joint_info[1].decode("utf-8") == target_link_name:
                    # The link ID is stored in the second element of the joint_info tuple
                    self.target_link_id = joint_info[0]
                    print(f"The ID of the link '{target_link_name}' is {self.target_link_id}")
                    break
            while(self.check_distance_to_obstacles(robot_id)):
                pos = self.get_random_robot_position()
                p.removeBody(robot_id)
                robot_id = p.loadURDF("turtlebot.urdf", pos)                
            self.robot_ids.append(robot_id)
            self.prev_pos.append(pos[:2])
        self.range_measurements_history = {}
        self.range_measurements_history = {robot_id: deque(maxlen=self.max_history) for robot_id in self.robot_ids}
        plane_id = p.loadURDF("plane.urdf")
        self.iter = 0

        # Load obstacles
        for _ in range(self.num_obstacles):
            
            obstacle_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height = 0.4)
            obstacle_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length = 0.4)
            obstacle_pos = self.get_random_obstacle_position()
            obstacle_pos[2] = 0.2
            obstacle_id = p.createMultiBody(10, obstacle_shape, obstacle_visual,basePosition=obstacle_pos)
            while(self.check_distance_to_obstacles(obstacle_id)):
                obstacle_pos = self.get_random_obstacle_position()
                obstacle_pos[2] = 0.2
                p.removeBody(obstacle_id)
                obstacle_id = p.createMultiBody(10, obstacle_shape, obstacle_visual,basePosition=obstacle_pos)

            self.obstacle_ids.append(obstacle_id)

        # Assign collision-free goal configurations to robots
        for robot_id in self.robot_ids:
            goal_position = self.get_random_goal_position()
            self.goal_positions.append(goal_position)
            object_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height = 0)
            object_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.1, length = 0.1)
            object_id = p.createMultiBody(0, 0, object_visual, basePosition=goal_position)
            self.commanded_vel.append((0,0))


        # TODO: Implement logic to set the goal positions for each robot
        # You can use p.createConstraint to set a constraint between the robot and its goal

        # Return initial observation
        return self._get_observation()

    def step(self, actions):
        # Apply actions for each robot
        self.commanded_vel = []
        for robot_id, action in zip(self.robot_ids,actions):

            forward_velocity, angular_velocity = action
            right_wheel_velocity = (2 * forward_velocity + angular_velocity * self.wheel_distance) / (2*self.wheel_radius )
            left_wheel_velocity = (2 * forward_velocity - angular_velocity * self.wheel_distance) / (2*self.wheel_radius)
            self.commanded_vel.append(action)
            p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=left_wheel_velocity)
            p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=right_wheel_velocity)

        for _ in range(24):
            p.stepSimulation()
            
        rewards = []
        self.current_pos = []
        dones = []
        truncated = []
        self.iter +=1
        for robot_id in self.robot_ids:
            dones.append(False)
            truncated.append(False)
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            self.current_pos.append(pos[:2])
            reward = self.compute_reward(robot_id)
            rewards.append(reward)
            if self.calculate_distance_to_goal(robot_id)[0] <= self.dist_tresh or self.min_distance[robot_id]<=0.05:
                dones[robot_id] = True
            if self.iter>=self.max_duration:
                truncated[robot_id] = True 
        # TODO: Simulate the environment for a time step
        rewards = np.stack(rewards,axis=0)
        dones = np.stack(dones,axis=0)
        truncated = np.stack(truncated,axis=0)
        self.prev_pos = self.current_pos
        # Return the next observation, reward, done, and info
        return self._get_observation(), rewards, dones, truncated

    def _get_observation(self):
        p.removeAllUserDebugItems()
        # Get range measurements for each robot using raycasting
        range_measurements = []
        for robot_id in self.robot_ids:
            range_measurement = self.get_range_measurements(robot_id)
            self.range_measurements_history[robot_id].append(range_measurement)

            # Pad with the oldest measurement if there are less than three measurements
            while len(self.range_measurements_history[robot_id]) < self.max_history:
                self.range_measurements_history[robot_id].appendleft(self.range_measurements_history[robot_id][0])

            range_measurements.append(np.array(self.range_measurements_history[robot_id]))

        # # Stack the range measurements
        # range_measurements = torch.stack(range_measurements)
        # Get other information (distance to goal, forward velocity, angular velocity) for each robot
        self.goal_distance = []
        observation = []
        for robot_id in self.robot_ids:
            distance_to_goal = self.calculate_distance_to_goal(robot_id)  # Replace with your distance calculation
            self.goal_distance.append(distance_to_goal)
            robot_velocity = self.commanded_vel[robot_id]  # Replace with your velocity calculation
            single_obs = np.concatenate((distance_to_goal, robot_velocity))
            single_obs = np.tile(single_obs, (1, 512)).reshape(512,4).T
            single_obs = np.vstack((range_measurements[robot_id], single_obs))
            observation.append(single_obs)
        observation = np.stack(observation, axis=0)
        self.compute_running_stats(observation)
        # Stack the other information
        return observation
    def visualize_rays(self, results, ray_from, ray_to):
        # Loop through each ray and add debug lines
        for i in range(len(ray_from)):
            if results[i]<1.0: # Red color for the rays
                ray_color = [1, 0, 0] 
                p.addUserDebugLine(ray_from[i], ray_to[i], ray_color, parentObjectUniqueId=-1, parentLinkIndex=-1)
    
    # Function to set up the environment and retrieve range measurements using ray casting
    def get_range_measurements(self, robot_id):
        # Use rayTestBatch to get range measurements
        num_rays = 512
        ray_from = np.zeros((num_rays, 3))
        ray_to = np.zeros((num_rays, 3))
        ray_directions = np.zeros((num_rays, 3))

        # Get the robot's position and orientation
        robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
        sensor_pos, sensor_orn = p.getLinkState(robot_id, self.target_link_id)[0], p.getLinkState(robot_id, self.target_link_id)[1]
        
        # Set up rays from the robot sensor
        for i in range(num_rays):
            # Replace 1 with the index of your sensor link
            
            # Incorporate robot orientation (yaw) into ray direction
            angle = (i / num_rays) * np.pi
            angles = np.array(p.getEulerFromQuaternion(robot_orn))
            ray_direction = np.array([np.cos(angle-np.pi/2+angles[2]), np.sin(angle-np.pi/2+angles[2]), 0.0])
            ray_from[i] = sensor_pos
            ray_to[i] = sensor_pos + 3.0 * ray_direction
            ray_directions[i] = ray_direction
        # Perform rayTestBatch
        results = p.rayTestBatch(ray_from, ray_to, numThreads=5)
        # Visualize rays


        # Extract range measurements
        range_measurements = [result[2] for result in results]
        self.min_distance[robot_id] = np.min(range_measurements)*3.0
        # self.update_range_stats(range_measurements)
        return np.array(range_measurements)

    
    # Function to get the linear and angular velocities of the robot
    def get_robot_velocity(self,robot_id):
        # Get the linear and angular velocities of the robot
        base_velocity = p.getBaseVelocity(robot_id)
        # Get the pose and velocity of the robot's base in the world frame
        base_pose =  p.getBasePositionAndOrientation(robot_id)

        # Convert the linear and angular velocity to the local frame
        rotation_matrix = p.getMatrixFromQuaternion(base_pose[1])  # Rotation matrix from quaternion
        rotation_matrix = np.reshape(rotation_matrix, (3, 3))

        # Linear velocity in local frame
        local_linear_velocity = np.dot(np.linalg.inv(rotation_matrix), np.array(base_velocity[0]))

        # Angular velocity in local frame
        local_angular_velocity = np.dot(np.linalg.inv(rotation_matrix), np.array(base_velocity[1]))
        # Extract the velocity values
        linear_velocity = np.array(local_linear_velocity)[0]
        angular_velocity = np.array(local_angular_velocity)[2]

        return np.array([linear_velocity, angular_velocity])

    def render(self, mode='human'):
        # Render the simulation
        p.stepSimulation()

    def close(self):
        # Close the PyBullet connection
        p.disconnect()

    def get_random_robot_position(self):
        # TODO: Implement logic to generate random positions for robots
        return [np.random.uniform(-4, 4), np.random.uniform(-4, 4), 0]

    def get_random_obstacle_position(self):
        # TODO: Implement logic to generate random positions for obstacles
        return [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0]

    def get_random_goal_position(self):
        # TODO: Implement logic to generate random goal positions
        return [np.random.uniform(-4, 4), np.random.uniform(-4, 4), 1.0]

    def check_distance_to_obstacles(self, robot_id):
        # TODO: Implement logic to check collision with obstacles
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        for other in self.robot_ids:
            if(other != robot_id):
                pr, _ = p.getBasePositionAndOrientation(other)
                dist = np.linalg.norm(np.array(pos[:2])-np.array(pr[:2]))
                if(dist<3*self.radius):
                    return True
        for obstacle in self.obstacle_ids:
            pr, _ = p.getBasePositionAndOrientation(obstacle)
            dist = np.linalg.norm(np.array(pos[:2])-np.array(pr[:2]))
            if(dist<2*self.radius):
                return True
        return False  

    def calculate_distance_to_goal(self, robot_id):
        pos, ori = p.getBasePositionAndOrientation(robot_id)
        goal = self.goal_positions[robot_id]
        # Calculate the vector from the robot to the goal
        vector_to_goal = np.array([(goal[0] - pos[0]), (goal[1] - pos[1])])
        _,_,yaw = p.getEulerFromQuaternion(ori)
        # Calculate the polar coordinates (distance, angle) of the vector
        distance = np.linalg.norm(vector_to_goal)
        angle = np.arctan2(vector_to_goal[1], vector_to_goal[0])-yaw

        return distance, angle

    def calculate_increment_to_goal(self, robot_id):
        old_distance, _ = np.array(self.goal_distance[robot_id])
        new_distance, _ = self.calculate_distance_to_goal(robot_id)
        return old_distance - new_distance
    
    def compute_reward(self, robot_id):
        self.get_range_measurements(robot_id)
        rcoll = self.rcoll if self.min_distance[robot_id]<=0.05 else 0
        rgoal = self.rarrival if self.calculate_distance_to_goal(robot_id)[0] <= self.dist_tresh else self.wgoal * self.calculate_increment_to_goal(robot_id)
        # Extract the velocity values
        angular_velocity = self.commanded_vel[robot_id][1]

        rw = self.wpenalty*np.abs(angular_velocity) if np.abs(angular_velocity)>0.7 else 0

        return rcoll+rgoal+rw
