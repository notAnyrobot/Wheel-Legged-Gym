import os
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RUNS = 10000
NUM_OBS = 27  # + 3
LEN_HISTORY = 5


# Load the JIT module
jit_model = torch.jit.load(os.path.join(SCRIPT_DIR, "model/policy(1).pt"))
jit_model.eval()

# Print the computational graph
# print(jit_model.graph)
total_time = 0

with torch.no_grad():
    for _ in range(RUNS):
        # Measure inference time
        start_time = time.time()

        # Prepare the input tensor
        observations = torch.rand(1, NUM_OBS)
        observation_history = torch.rand(1, LEN_HISTORY, NUM_OBS)

        # Perform inference
        actions = jit_model(observations, observation_history.view(1, -1))
        end_time = time.time()

        # Calculate and accumulate inference time
        inference_time = end_time - start_time
        total_time += inference_time

        # Print the output
        # print(actions.shape)

# Calculate and print the average inference time
average_inference_time = total_time / RUNS
print(f"Total inference time: {total_time:.6f} seconds for {RUNS} runs")
print(f"Average inference time: {average_inference_time:.6f} seconds")
