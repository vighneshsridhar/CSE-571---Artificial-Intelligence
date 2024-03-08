import pandas as pd
import os
import shutil

network_params = pd.read_csv('normalized_data.csv')
params_collision_0 = network_params.loc[network_params.iloc[:, 6] == 0]
params_collision_1 = network_params.loc[network_params.iloc[:, 6] == 1]
total_samples = len(network_params)
required_samples = 4000
split = 0.6

if (params_collision_0.shape[0] >= split * required_samples):
    params_collision_0 = params_collision_0.sample(n=int(split * required_samples))

if (params_collision_1.shape[0] >= (1 - split) * required_samples):
    params_collision_1 = params_collision_1.sample(n=int(((1-split)*required_samples)))

print (len(params_collision_0))
print (len(params_collision_1))

params_collision = pd.concat([params_collision_0, params_collision_1], ignore_index = True)
params_collision = params_collision.sample(frac = 1).reset_index(drop = True)
params_collision.to_csv('../assignment_part4/saved/training_data.csv', header = False, index = False)
#cwd = os.getcwd()
#source_file = os.path.join(cwd, 'filtered_train_data.csv')
#destination_file = os.path.join(os.path.basename(cwd), "assignment_part4", "saved", "training_data.csv")
#shutil.copy(source_file, destination_file)
