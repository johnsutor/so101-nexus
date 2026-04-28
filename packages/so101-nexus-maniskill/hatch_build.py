"""Hatchling metadata hook to pin so101-nexus-core to the package version."""

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    """Pin so101-nexus-core to the package version."""

    def update(self, metadata: dict) -> None:
        """Set dependencies with so101-nexus-core pinned to this package's version."""
        version = metadata["version"]
        metadata["dependencies"] = [
            f"so101-nexus-core=={version}",
            "mani_skill>=3.0.1",
            "setuptools<82",
        ]
        metadata["optional-dependencies"] = {
            "teleop": [f"so101-nexus-core[teleop]=={version}"],
        }
