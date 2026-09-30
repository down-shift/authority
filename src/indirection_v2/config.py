"""Generate and validate the frozen v2 matched dataset."""
from .worlds import generate_worlds
from .render import examples,K_DEFAULT
from .validation import audit

def generate(config):
    k=int(config.get("total_relation_edges",K_DEFAULT))
    seed=int(config["seed"])
    worlds=generate_worlds(int(config["worlds_per_family"]),seed)
    rows=examples(worlds,k,seed)
    report=audit(rows,k,seed)
    expected=int(config["worlds_per_family"])*3*19
    if len(rows)!=expected: raise AssertionError(f"expected {expected} v2 rows, got {len(rows)}")
    return rows,report
