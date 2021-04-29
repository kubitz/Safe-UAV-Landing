from enum import Enum

from safelanding.utils import bidict

labelsAero = {
    "background": 0,
    "person": 1,
    "bike": 2,
    "car": 3,
    "drone": 4,
    "boat": 5,
    "animal": 6,
    "obstacle": 7,
    "construction": 8,
    "vegetation": 9,
    "road": 10,
    "sky": 11,
}

labelsGraz = {
    "unlabeled": 0,
    "pavedArea": 1,
    "dirt": 2,
    "grass": 3,
    "dirt": 4,
    "water": 5,
    "rocks": 6,
    "pool": 7,
    "lowVegetation": 8,
    "roof": 9,
    "wall": 10,
    "window": 11,
    "door": 12,
    "fence": 13,
    "fencePole": 14,
    "person": 15,
    "animal": 16,
    "car": 17,
    "bike": 18,
    "tree": 19,
    "baldTree": 20,
    "arMarker": 21,
    "obstacle": 22,
    "conflicting": 23,
}


class RiskLevel(Enum):
    VERY_HIGH = 100
    HIGH = 20
    MEDIUM = 10
    LOW = 5
    ZERO = 0


# Risk table for the safe landing zone finder
risk_table = {
    "unlabeled": RiskLevel.ZERO,
    "pavedArea": RiskLevel.LOW,
    "dirt": RiskLevel.LOW,
    "grass": RiskLevel.ZERO,
    "dirt": RiskLevel.LOW,
    "water": RiskLevel.HIGH,
    "rocks": RiskLevel.MEDIUM,
    "pool": RiskLevel.HIGH,
    "lowVegetation": RiskLevel.ZERO,
    "roof": RiskLevel.HIGH,
    "wall": RiskLevel.HIGH,
    "window": RiskLevel.HIGH,
    "door": RiskLevel.HIGH,
    "fence": RiskLevel.HIGH,
    "fencePole": RiskLevel.HIGH,
    "person": RiskLevel.VERY_HIGH,
    "animal": RiskLevel.VERY_HIGH,
    "car": RiskLevel.VERY_HIGH,
    "bike": RiskLevel.VERY_HIGH,
    "tree": RiskLevel.HIGH,
    "baldTree": RiskLevel.HIGH,
    "arMarker": RiskLevel.ZERO,
    "obstacle": RiskLevel.HIGH,
    "conflicting": RiskLevel.HIGH,
    "background": RiskLevel.ZERO,
    "drone": RiskLevel.MEDIUM,
    "boat": RiskLevel.MEDIUM,
    "construction": RiskLevel.HIGH,
    "vegetation": RiskLevel.LOW,
    "road": RiskLevel.ZERO,
    "sky": RiskLevel.VERY_HIGH,
}

# Safety indicator for the metrics module. This is the Ground truth.
notSafe = [
    "water",
    "fence",
    "fencePole",
    "animal",
    "car",
    "bicycle",
    "tree",
    "baldTree" "obstacle",
    "pool",
    "wall",
    "door",
    "roof",
    "drone",
    "construction",
    "boat",
    "obstacle",
]

biLabelsAero = bidict(labelsAero)
biLabelsGraz = bidict(labelsGraz)

datasetLabels = {"aeroscapes": biLabelsAero, "graz": biLabelsGraz}
