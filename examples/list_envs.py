import so101_nexus.mujoco
import so101_nexus.warp  # noqa: F401  (registers Warp envs)
from so101_nexus.env_ids import all_registered_env_ids

if __name__ == "__main__":
    for env_id in all_registered_env_ids():
        print(env_id)
