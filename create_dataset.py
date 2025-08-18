import datetime

import numpy as np
import osu.dataset as dataset

REPLAY_LIMIT = 2000
mrekk_id = 7562902
obj_dataset = dataset.user_replay_mapping_from_cache(user_id=mrekk_id, limit=REPLAY_LIMIT)

print(len(obj_dataset))

input_data = dataset.input_data(obj_dataset, verbose=True)
output_data = dataset.target_data(obj_dataset, verbose=True)

input = input_data
output = output_data

xs = np.reshape(input.values, (-1, dataset.BATCH_LENGTH, len(dataset.INPUT_FEATURES)))
ys = np.reshape(output.values, (-1, dataset.BATCH_LENGTH, len(dataset.OUTPUT_FEATURES)))

timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")

np.save(f'.datasets/xs_{len(obj_dataset)}_{timestamp}.npy', xs)
np.save(f'.datasets/ys_{len(obj_dataset)}_{timestamp}.npy', ys)