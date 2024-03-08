# ************** STUDENTS EDIT THIS FILE **************

from SteeringBehaviors import Wander
import SimulationEnvironment as sim

import numpy as np
import pandas as pd

def collect_training_data(total_actions):
    #set-up environment
    sim_env = sim.SimulationEnvironment()

    #robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    num_params = 7
    #STUDENTS: network_params will be used to store your training data
    # a single sample will be comprised of: sensor_readings, action, collision
    network_params = pd.DataFrame(columns = ["Sensor Reading 1", "Sensor Reading 2", "Sensor Reading 3", "Sensor Reading 4",
    "Sensor Reading 5", "Action", "Collision"])
    network_params = pd.concat([network_params, pd.DataFrame({"Sensor Reading 1": [0], "Sensor Reading 2": [0], "Sensor Reading 3": [0], "Sensor Reading 4": [0], "Sensor Reading 5": [0], "Action": [0], "Collision": [0]})], ignore_index=True)


    for action_i in range(total_actions):
        progress = 100*float(action_i)/total_actions
        print(f'Collecting Training Data {progress}%   ', end="\r", flush=True)

        #steering_force is used for robot control only
        action, steering_force = steering_behavior.get_action(action_i, sim_env.robot.body.angle)

        for action_timestep in range(action_repeat):
            if action_timestep == 0:
                _, collision, sensor_readings = sim_env.step(steering_force)
            else:
                _, collision, sensor_readings = sim_env.step(steering_force)

            if collision:
                steering_behavior.reset_action()
                #STUDENTS NOTE: this statement only EDITS collision of PREVIOUS action
                #if current action is very new.
                if action_timestep < action_repeat * .3: #in case prior action caused collision
                    #network_params[-1][-1] = collision #share collision result with prior action
                    network_params.iat[-1, -1] = collision
                break


        #STUDENTS: Update network_params.

        temp = pd.DataFrame({"Sensor Reading 1": [sensor_readings[0]], "Sensor Reading 2": [sensor_readings[1]],
        "Sensor Reading 3": [sensor_readings[2]], "Sensor Reading 4": [sensor_readings[3]], "Sensor Reading 5": [sensor_readings[4]]
        , "Action": [action], "Collision": [collision]})
        network_params = pd.concat([network_params, temp], ignore_index = True)


    #STUDENTS: Save .csv here. Remember rows are individual samples, the first 5
    #columns are sensor values, the 6th is the action, and the 7th is collision.
    #Do not title the columns. Your .csv should look like the provided sample.

    network_params.to_csv('train_data.csv', header = False, index = False)






if __name__ == '__main__':
    # total_actions = 4000
    total_actions = 10000
    collect_training_data(total_actions)
