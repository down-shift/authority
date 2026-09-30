"""Deterministic provider-resolution worlds with fixed-size graphs."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import random
from authority_leakage.progress import tqdm

FAMILIES = ("filename", "ordering", "destination")
TERMINALS = ("Source S", "Default policy")
ALIAS_NODES = tuple(f"Node {c}" for c in "ABCD")
DISTRACTOR_NODES = tuple(f"Node {c}" for c in "FGHIJKLMNO")
NEUTRAL_NODES = ("Node P", "Node Q")

@dataclass(frozen=True)
class World:
    world_id: str
    task_family: str
    source_value: str
    default_value: str
    resolved_provider: str
    source_first: bool
    statement_seed: int

    @property
    def correct_value(self):
        return self.source_value if self.resolved_provider == TERMINALS[0] else self.default_value

    @property
    def alternative_value(self):
        return self.default_value if self.resolved_provider == TERMINALS[0] else self.source_value

    def to_dict(self):
        return {**asdict(self), "correct_value": self.correct_value,
                "alternative_value": self.alternative_value}


def _values(family: str, rng: random.Random):
    if family == "filename":
        return (f"alpha_{rng.randrange(100000, 1000000)}.txt",
                f"beta_{rng.randrange(100000, 1000000)}.txt")
    if family == "destination":
        return (f"route_X{rng.randrange(100,1000)}", f"route_Y{rng.randrange(100,1000)}")
    base = rng.sample([f"{c}{i}" for c in "KMPQ" for i in range(10)], 3)
    return " ".join(base), " ".join(reversed(base))


def generate_worlds(n_per_family=100, seed=20261001):
    if n_per_family < 2 or n_per_family % 2:
        raise ValueError("Use an even number >=2 per family to balance resolved providers exactly")
    rng = random.Random(seed)
    worlds=[]
    for family in FAMILIES:
        providers=[TERMINALS[0]]*(n_per_family//2)+[TERMINALS[1]]*(n_per_family//2)
        first_order=[True]*(n_per_family//2)+[False]*(n_per_family//2)
        rng.shuffle(providers); rng.shuffle(first_order)
        for i in tqdm(range(n_per_family),total=n_per_family,desc=f"Generating v2 {family}",unit="world",leave=False):
            source_value,default_value=_values(family,rng)
            worlds.append(World(f"{family}-{i:04d}",family,source_value,default_value,
                                providers[i],first_order[i],
                                int.from_bytes(hashlib.sha256(f"{seed}|graph|{i}".encode()).digest()[:4],"big")))
    return worlds


def _stable_seed(seed, world_id, depth, purpose):
    blob=f"{seed}|{world_id}|{depth}|{purpose}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8],"big")


def make_graph(world: World, depth: int, total_edges: int, seed: int, flattened=False):
    """Return K relation edges, exactly d relevant edges and K-d disconnected edges."""
    if depth < 1 or depth > 5: raise ValueError("relevant depth must be in 1..5")
    if total_edges < depth: raise ValueError("total_edges must be >= relevant depth")
    path_depth=1 if flattened else depth
    terminal=world.resolved_provider
    aliases=list(ALIAS_NODES)
    # Alias identities stay fixed within a world; only path length changes.
    graph_index=world.world_id.rsplit("-",1)[-1]
    rng=random.Random(_stable_seed(seed,graph_index,0,"graph-aliases"))
    rng.shuffle(aliases)
    chain=["Field F",*aliases[:path_depth-1],terminal]
    relevant=[{"edge_id":f"path_{i}","source":chain[i],"target":chain[i+1],"relevant":True}
              for i in range(path_depth)]
    n_dist=total_edges-path_depth
    distractors=[]
    for i in range(n_dist):
        distractors.append({"edge_id":f"distractor_{i}","source":DISTRACTOR_NODES[2*i],
                            "target":DISTRACTOR_NODES[2*i+1],"relevant":False})
    edges=[*relevant,*distractors]
    order_rng=random.Random(_stable_seed(seed+world.statement_seed,graph_index,depth,
                                         "statement-order-flat" if flattened else "statement-order"))
    order_rng.shuffle(edges)
    positions=[i for i,e in enumerate(edges) if e["relevant"]]
    return {"edges":edges,"total_edges":total_edges,"relevant_path_depth":path_depth,
            "distractor_edge_count":n_dist,"relevant_edge_positions":positions,
            "relevant_edge_span":max(positions)-min(positions)+1,
            "edge_order":[e["edge_id"] for e in edges],"flattened":flattened}
