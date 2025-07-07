from gym import error, spaces, utils
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from ray.rllib.models import ModelCatalog
import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch

ACTION_VEC_SIZE = 1

action_encoder = ModelCatalog.get_preprocessor_for_space( 
                                                            # Box(-np.inf, np.inf, (ACTION_VEC_SIZE,), np.float32) # Unbounded
                                                        #  spaces.MultiDiscrete([3,3]) 
                                                        # spaces.Discrete(3),
                                                        Box(0, 2, (ACTION_VEC_SIZE,), np.int16)
                                                            )
# _, opponent_batch = original_batches[other_id]
opponent_actions = np.array([action_encoder.transform(1)], dtype=np.int32) # Unbounded
print(opponent_actions)
# opponent_actions = np.array([action_encoder.transform(np.clip(a, -1, 1)) for a in opponent_batch[SampleBatch.ACTIONS]]) # Bounded
# to_update[:, -ACTION_VEC_SIZE:] = opponent_actions