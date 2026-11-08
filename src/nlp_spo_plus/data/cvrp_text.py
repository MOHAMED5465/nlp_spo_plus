import math
import numpy as np
from scipy.spatial import distance

# definition of latent state penalties (additive edge cost penalties)
WEATHER_STATES = {
    "clear": 0.0,
    "rain": 0.5,
    "snow": 1.2
}

TRAFFIC_STATES = {
    "smooth": 0.0,
    "heavy": 0.6,
    "jammed": 1.5
}

EVENT_STATES = {
    "none": 0.0,
    "accident": 1.0,
    "roadworks": 0.8
}

def genLatentData(
    num_data: int,
    num_clients: int,
    num_vehicules: int = 1,
    deg: int = 1,
    noise_width: float = 0,
    seed: int = 42
) -> tuple[list[dict], np.ndarray, np.ndarray, int]:
    """
    generates cvrp instances with node-level latent state features.
    
    returns:
        latent_features (list of dict): node states (weather, traffic, event) per instance
        costs (np.ndarray): true cost matrices (c*) for all instances
        demands (np.ndarray): customer node demands
        capacity (int): vehicle capacity limit
    """
    if deg < 1:
        raise ValueError(f"polynomial degree must be >= 1, got {deg}")
    if noise_width < 0:
        raise ValueError(f"noise_width must be non-negative, got {noise_width}")

    rnd = np.random.RandomState(seed)
    n = num_data
    m = num_clients + 1
    
    # base spatial node coordinates generation
    coords = np.concatenate((rnd.uniform(-2, 2, (m // 2, 2)), rnd.normal(0, 1, (m - m // 2, 2))))
    dist_matrix = distance.cdist(coords, coords, "euclidean")
    
    latent_features_all = []
    costs = np.zeros((n, m * (m - 1)), dtype=np.float32)
    
    # pre-compute noise matrix for all instances and edges
    noise = rnd.uniform(1 - noise_width, 1 + noise_width, (n, m * (m - 1)))
    
    for instance_idx in range(n):
        # sample latent states for every node including depot (0)
        instance_nodes_states = []
        for node in range(m):
            weather = rnd.choice(list(WEATHER_STATES.keys()))
            traffic = rnd.choice(list(TRAFFIC_STATES.keys()))
            event = rnd.choice(list(EVENT_STATES.keys()))
            
            instance_nodes_states.append({
                "weather": weather,
                "traffic": traffic,
                "event": event
            })
            
        latent_features_all.append(instance_nodes_states)
        
        # calculate non-linear true cost matrix (ground truth)
        c_instance = np.zeros(m * (m - 1))
        edge_idx = 0
        for i in range(m):
            for j in range(m):
                if i != j:
                    # total penalty combines source (i) and target (j) node penalties
                    penalty_i = (
                        WEATHER_STATES[instance_nodes_states[i]["weather"]] +
                        TRAFFIC_STATES[instance_nodes_states[i]["traffic"]] +
                        EVENT_STATES[instance_nodes_states[i]["event"]]
                    )
                    penalty_j = (
                        WEATHER_STATES[instance_nodes_states[j]["weather"]] +
                        TRAFFIC_STATES[instance_nodes_states[j]["traffic"]] +
                        EVENT_STATES[instance_nodes_states[j]["event"]]
                    )
                    
                    # align feature cost calculation with numerical baseline structure
                    feature_cost = (((penalty_i + penalty_j) + 3) ** deg) / 3 ** (deg - 1)
                    true_cost = dist_matrix[i, j] + feature_cost * noise[instance_idx, edge_idx]
                    
                    c_instance[edge_idx] = true_cost
                    edge_idx += 1
                    
        costs[instance_idx] = np.around(c_instance, decimals=4)
        
    # demands and capacity calculation
    demands = np.zeros(m, dtype=int)
    demands[1:] = rnd.randint(1, 10, size=num_clients)
    capacity = math.ceil((np.sum(demands) / num_vehicules) * 1.2)
    
    return latent_features_all, costs, demands, int(capacity)


# prompt builder for llm text generation
def build_node_prompt(node_state: dict) -> str:
    """
    constructs the prompt sent to the llm generator for a specific node.
    """
    weather = node_state["weather"]
    traffic = node_state["traffic"]
    event = node_state["event"]
    
    prompt = (
        "You are a local traffic reporter or city news reporter. "
        "Write a very short news update (2 to 3 sentences maximum) describing "
        "the current situation in a specific city district.\n\n"
        "MANDATORY CONSTRAINTS:\n"
        f"- Weather condition: {weather}.\n"
        f"- Road traffic status: {traffic}.\n"
        f"- Ongoing event: {event}.\n"
        "- You MUST include ONE sentence containing a distractor fact or local anecdote "
        "(e.g., a local soccer match result, an upcoming election, a concert, or a bakery opening) "
        "that has ABSOLUTELY NO IMPACT on delivery truck traffic or routing costs.\n\n"
        "Do not use bullet points or lists; write natural and fluent prose."
    )
    return prompt