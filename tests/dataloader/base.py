"""
Agent class -- provides an interface for loading agent data
Simulations may preload the data for the agent as well; as long as these
base interfaces are implemented, the extended agent class should be 
compatible with the analyses. The agent class is a read-only data 
structure (from a requirements perspective) but different simulation 
engines may have some write extensions beyond the base class if needed.
"""
class Agent:
    def __init__(self): 
        self.id = None
        self._data = {}
        self._stop_points = {}

    """
    data format: (3, N) pandas dataframe with columns "lat", 
    "long" and "time" -- formats are float, float and timestamp
    """
    def data(self, context="past"):
        if context in self._data:
            return self._data[context]
        else:
            # TODO: yield a warning to the user
            return None

    """
    data format: (4, N) pandas dataframe with columns "lat", "long", 
    "start" and "end" -- types are 2xfloat, 2xtimestamp
    """
    def stop_points(self, context="past"):
        if context in self._stop_points:
            return self._stop_points[context]
        else:
            # TODO: yield a warning to the user
            return None

class Simulation:
    def __init__(self):
        pass

    """
    Returns an Agent object given an agent id
    """
    def agent(self, agent_id):
        pass

    
    """
    Returns a structured set of agents optimized for high performance
    """
    def iteragents(self):
        pass
