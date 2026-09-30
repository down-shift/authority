"""Generate semantics once; render repeated measurements from frozen worlds."""
from dataclasses import asdict, dataclass
import hashlib
import json
import random

FAMILIES = ("filename", "ordering", "destination")

@dataclass(frozen=True)
class World:
    world_id: str
    task_family: str
    source: str
    default: str
    owner: str
    role: str
    field_symbol: str
    source_first: bool
    source_value: str
    default_value: str
    context: str

    @property
    def correct_value(self):
        return self.source_value if self.owner == self.source else self.default_value

    @property
    def alternative_value(self):
        return self.default_value if self.owner == self.source else self.source_value

    def to_dict(self):
        return {**asdict(self), "correct_value": self.correct_value, "alternative_value": self.alternative_value}


def generate_worlds(n_per_family=100, seed=20260930):
    if n_per_family < 2 or n_per_family % 2:
        raise ValueError("Use an even number >=2 per family for exact owner counterbalancing")
    rng = random.Random(seed)
    worlds = []
    for family in FAMILIES:
        owners = [True] * (n_per_family // 2) + [False] * (n_per_family // 2)
        rng.shuffle(owners)
        source_first_order = [True] * (n_per_family // 2) + [False] * (n_per_family // 2)
        rng.shuffle(source_first_order)
        for i, source_correct in enumerate(owners):
            source = "Source " + rng.choice(["S", "T", "U", "V"])
            role = "R" + str(rng.randrange(100, 1000))
            field = "F" + str(rng.randrange(100, 1000))
            if family == "filename":
                values = [f"{p}_{rng.randrange(100000, 1000000)}.txt" for p in ("filex", "filey")]
                context = ""
            elif family == "destination":
                values = [f"route_{p}{rng.randrange(100,1000)}" for p in ("X", "Y")]
                context = ""
            else:
                items = rng.sample([f"{c}{j}" for c in "KMPQ" for j in range(10)], 3)
                values = [" ".join(items), " ".join(reversed(items))]
                context = "Base sequence: " + " ".join(items) + ". Proposals are literal final sequences."
            rng.shuffle(values)
            worlds.append(World(f"{family}-{i:04d}", family, source, "Default policy",
                                source if source_correct else "Default policy", role, field, source_first_order[i],
                                values[0], values[1], context))
    return worlds


def dataset_sha256(rows):
    payload = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    return hashlib.sha256(payload.encode()).hexdigest()
