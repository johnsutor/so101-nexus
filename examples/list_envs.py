from so101_nexus_core.env_ids import all_registered_env_ids

if __name__ == "__main__":
    for env_id in all_registered_env_ids():
        print(env_id)
