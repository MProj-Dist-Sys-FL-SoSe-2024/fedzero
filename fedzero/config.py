import gurobipy

NIID_DATA_SEED = 42  # controls how the data is split across clients
SAVE_TRAINED_MODELS = False
GUROBI_ENV = gurobipy.Env(params={"OutputFlag": 0})

TIMESTEP_IN_MIN = 1  # minutes
MAX_ROUND_IN_MIN = 60  # minutes
MAX_ROUNDS = 120
MAX_TIME_IN_DAYS = 7  # currently 11 max
STOPPING_CRITERIA = None  # rounds without improved accuracy

ENABLE_BROWN_CLIENTS = False
TIME_WINDOW_LOWER_BOUND = 101
TIME_WINDOW_UPPER_BOUND = 120
BROWN_CLIENTS_BUDGET_PERCENTAGE = 4.0
BROWN_CLIENTS_NUMBER_PERCENTAGE = 4.0
BROWN_EXCLUSION_UPDATE = False

DATA_SUBSET = 1.0

NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
BATCH_SIZE = 10
MIN_LOCAL_EPOCHS = 1
MAX_LOCAL_EPOCHS = 5

SOLAR_SIZE = 800  # W

# Flower
RAY_CLIENT_RESOURCES = {
    "num_cpus": 1,  # CPU threads assigned to each client
    # "num_gpus": 1 / 3
}
RAY_INIT_ARGS = {
    "num_cpus": 8,  # Number of physically accessible CPUs
    # "num_gpus": 1,  # Number of physically accessible GPUs
    "ignore_reinit_error": True,
    "include_dashboard": True,
    # "object_store_memory": 20*1024*1024*1024,  # Reduced to 30GB
    # "memory": 80*1024*1024*1024,  # Allocating 80GB for tasks and actors, adjust as needed
}
