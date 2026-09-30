from .worlds import generate_worlds
from .render import examples
from .validation import audit

def generate(config):
    rows=examples(generate_worlds(int(config['worlds_per_family']),int(config['seed'])))
    report=audit(rows)
    if len(rows)!=int(config['worlds_per_family'])*3*5*2: raise AssertionError('Unexpected dataset size')
    return rows,report
