"""Placement metadata must not relabel existing episodes."""

import json

import pytest

from so101_nexus import PickAndPlaceV2Config
from so101_nexus.teleop.dataset import save_recorded_episode


class EpisodeStore:
    def __init__(self, root, *, fail=False):
        self.root = root
        self.num_episodes = 0
        self.fail = fail

    def save_episode(self):
        if self.fail:
            raise RuntimeError("The episode write failed.")
        self.num_episodes += 1


def test_contract_is_immutable_and_does_not_relabel_legacy_episode(tmp_path):
    dataset = EpisodeStore(tmp_path)
    save_recorded_episode(dataset)
    contract = PickAndPlaceV2Config(target_disc_radius=0.03).placement_contract
    save_recorded_episode(dataset, contract)
    directory = tmp_path / "meta" / "placement_contracts"
    assert not (directory / "episode_000000.json").exists()
    path = directory / "episode_000001.json"
    assert json.loads(path.read_text())["target_radius"] == 0.03
    dataset.num_episodes = 1
    with pytest.raises(FileExistsError):
        save_recorded_episode(dataset, PickAndPlaceV2Config().placement_contract)
    assert json.loads(path.read_text())["target_radius"] == 0.03
    assert dataset.num_episodes == 1


def test_failed_episode_does_not_leave_a_contract_for_an_unsaved_record(tmp_path):
    dataset = EpisodeStore(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="episode write failed"):
        save_recorded_episode(dataset, PickAndPlaceV2Config().placement_contract)
    assert not (tmp_path / "meta" / "placement_contracts" / "episode_000000.json").exists()
