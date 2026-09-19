"""Per-world deterministic random streams for batched Warp environments."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
import warp as wp


@wp.kernel
def _uniform_kernel(
    seeds: wp.array[wp.int64],
    episodes: wp.array[wp.int64],
    offsets: wp.array[wp.int64],
    worlds: wp.array[wp.int64],
    width: int,
    stream: int,
    out: wp.array[wp.float32],
):
    i = wp.tid()
    row = i // width
    world = worlds[row]
    draw = offsets[world] + wp.int64(i - row * width)
    episode = episodes[world]
    state = wp.rand_init(wp.int32(seeds[world]), wp.int32(0))
    state = wp.rand_init(wp.randi(state), wp.int32(episode & wp.int64(0xFFFFFFFF)))
    state = wp.rand_init(wp.randi(state), wp.int32(episode >> wp.int64(32)))
    state = wp.rand_init(wp.randi(state), wp.int32(stream))
    state = wp.rand_init(wp.randi(state), wp.int32(draw & wp.int64(0xFFFFFFFF)))
    state = wp.rand_init(wp.randi(state), wp.int32(draw >> wp.int64(32)))
    out[i] = wp.randf(state)


class WorldEpisodeRNG:
    """Independent PCG streams keyed by world, episode, and stream name."""

    _STREAMS = {"robot": 1, "task": 2, "wrist": 3, "visual": 4}

    def __init__(
        self,
        num_worlds: int,
        device: torch.device,
        seed: object | None,
    ) -> None:
        wp.init()
        self.device = device
        self.num_worlds = num_worlds
        self.seeds = torch.zeros(num_worlds, dtype=torch.int64, device=device)
        self.episodes = torch.zeros(num_worlds, dtype=torch.int64, device=device)
        self.offsets = torch.zeros(
            (len(self._STREAMS), num_worlds), dtype=torch.int64, device=device
        )
        self.seed(0 if seed is None else seed)

    def seed(self, seed: object) -> None:
        if isinstance(seed, bool):
            raise ValueError("seeds must be integers in [0, 2**32)")
        if isinstance(seed, Integral):
            values = [int(seed)] * self.num_worlds
        elif isinstance(seed, Sequence) and not isinstance(seed, str | bytes):
            values = list(seed)
        else:
            raise ValueError("seed must be an integer or a sequence of integers")
        if len(values) != self.num_worlds:
            raise ValueError(f"seed sequence length {len(values)} != num_envs {self.num_worlds}")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= int(value) < 2**32
            for value in values
        ):
            raise ValueError("seeds must be integers in [0, 2**32)")
        base = torch.tensor([int(value) for value in values], dtype=torch.int64, device=self.device)
        if isinstance(seed, Integral):
            base = (base + torch.arange(self.num_worlds, device=self.device) * 2654435761) % 2**32
        self.seeds.copy_(base)
        self.episodes.zero_()
        self.offsets.zero_()

    def begin(self, worlds: torch.Tensor, *, advance: bool) -> None:
        if advance:
            self.episodes[worlds] += 1
        self.offsets[:, worlds] = 0

    def rand(self, stream: str, worlds: torch.Tensor, *tail: int) -> torch.Tensor:
        stream_index = self._STREAMS[stream] - 1
        width = 1
        for size in tail:
            width *= size
        out = torch.empty((worlds.numel() * width,), device=self.device)
        wp.launch(
            _uniform_kernel,
            dim=out.numel(),
            inputs=[
                wp.from_torch(self.seeds),
                wp.from_torch(self.episodes),
                wp.from_torch(self.offsets[stream_index]),
                wp.from_torch(worlds),
                width,
                stream_index + 1,
            ],
            outputs=[wp.from_torch(out)],
            device=str(self.device),
        )
        self.offsets[stream_index, worlds] += width
        return out.reshape(worlds.numel(), *tail)
