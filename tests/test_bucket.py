import multiprocessing
from dataloader import noisy

import pandas as pd
import numpy as np

import argparse
import reepy

def process_agent(agent):
    def pd_to_np(dfs):
        df = pd.concat(dfs, axis=0, ignore_index=True)

        dates = df['timestamp'].dt.date
        df['time'] = (df['timestamp'].dt.hour * 3600 +
            df['timestamp'].dt.minute * 60 + df['timestamp'].dt.second) / (60 * 60 * 24)
        df['traj'] = pd.Categorical(dates).codes + 1
        return df[['traj', 'time', 'latitude', 'longitude']].to_numpy()

    trajectories = pd_to_np((agent.data("past"), 
                             # agent.data("future")
                             ))

    # rescaling time axis for reasonable results
    trajectories[:, 1] = 0
    
    agent_reeb = reepy.SparseReebGraph(dim=2, epsilon=1e-2)

    # agent_reeb.append_trajectories(trajectories)

    trajs = trajectories
    _, masks = np.unique(trajs[:, 0], return_index=True)
    trajs_split = np.split(trajs, masks[1:])

    for i, traj in enumerate(trajs_split):
        # remove the trajectory dimension
        print("inserting", i+1, "out of", len(trajs_split), "-- bundle count is", len(agent_reeb._bundles))
        agent_reeb.append_trajectory(traj[:, 1:])

    print("Building Reeb Graph...")

    agent_reeb.build()

    # print statistics (for our testing purposes)
    print("nodes:", agent_reeb.number_of_nodes(), 
          "edges:", agent_reeb.number_of_edges(),
          "samples:", trajectories.shape[0])


def process_bucket(bucket):
    sim1  = noisy.Simulation()

    # test with 750 agents
    for agent in range(int(bucket) * 1000, int(bucket) * 1000 + 750):
        print(agent)
        process_agent(sim1.agent(agent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bucket Processor")
    parser.add_argument("id", help="Bucket ID to process")

    args = parser.parse_args()

    print("Processing bucket", args.id)

    process_bucket(args.id)

    print("Finished bucket", args.id)
