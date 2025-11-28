from . import base
from pathlib import Path
import pandas as pd
import numpy as np

class Simulation(base.Simulation):
    # NOTE: Modified from base code
    def __init__(self, datapath="~/Datasets/NOISY/data.zstd.parquet"):
        self.datapath = \
            lambda context, bucket: Path(datapath).expanduser()
    
    def agent(self, agent_id, contexts=("past",)):
        bucket = agent_id // 1000

        agent = base.Agent()

        for context in contexts:
            agent._data[context] = pd.read_parquet(
                    self.datapath(context, bucket),
                    filters=[("agent", "==", agent_id)]
                )[["latitude", "longitude", "timestamp"]].rename({
                    "latitude": "lat",
                    "longitude": "long",
                    "timestamp": "time"
                }).reset_index(drop=True)

        return agent

    # returns a generator which contains chunks of agents
    def iteragents(self, bucket=0, chunks=50, contexts=("past", "future")):
        # loads in all agents in a single bucket
        # TODO: does it make sense to do this for multiple buckets?
        data = {}

        unique_agents = np.array([])

        for context in contexts:
            # TODO: stop points?
            data[context] = pd.read_parquet(
                self.datapath(context, bucket)
            )

            unique_agents = np.union1d(data[context]['agent'].unique(),
                                       unique_agents)

        groups = np.array_split(unique_agents, chunks)
        
        # note: this only supports up to 256 workers -- should be fine for most
        # applications but this may be a bottleneck later on
        worker_labels = np.empty(len(unique_agents), dtype=np.int8)
        for i, group_agents in enumerate(groups):
            indices = np.searchsorted(unique_agents, group_agents)
            worker_labels[indices] = i

        for context in contexts:
            data[context]["agent_worker"] = worker_labels[pd.Categorical(
                data[context]['agent'],
                categories=unique_agents,
                ordered=True).codes]

        def extract_worker_agents():
            for i in range(chunks):
                group = set(groups[i])

                agents = []
                group_data = {}

                # initialize agent id
                for agent_id in group:
                    agent = base.Agent()
                    agent.id = agent_id
                    agents.append(agent)

                for context in contexts:
                    group_data[context] = data[context][data[context]["agent"].isin(group)]

                    for j, agent in enumerate(agents):
                        agent._data[context] = group_data[context][
                            group_data[context]["agent"] == agent.id
                        ][["latitude", "longitude", "timestamp"]].rename({
                            "latitude": "lat",
                            "longitude": "long",
                            "timestamp": "time"
                        }).reset_index(drop=True)

                print(f"extracted {i} of {chunks} chunks for bucket {bucket}")

                yield agents

        return extract_worker_agents()
