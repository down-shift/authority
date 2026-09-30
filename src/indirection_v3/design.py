"""Factorial v3 worlds, prompts, and structural audits."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib, random

FAMILIES=("filename","ordering","destination")
TERMINALS=("Source S","Default policy")
DEPTHS=tuple(range(1,6)); DISTRACTORS=(0,2,4,6); LAYOUTS=("compact","dispersed")

@dataclass(frozen=True)
class World:
    world_id:str; task_family:str; source_value:str; default_value:str
    resolved_provider:str; seed:int
    @property
    def correct_value(self): return self.source_value if self.resolved_provider==TERMINALS[0] else self.default_value
    @property
    def alternative_value(self): return self.default_value if self.resolved_provider==TERMINALS[0] else self.source_value
    def to_dict(self): return {**asdict(self),"correct_value":self.correct_value,"alternative_value":self.alternative_value}

def _rng(seed,*parts):
    x="|".join(map(str,(seed,*parts))).encode(); return random.Random(int.from_bytes(hashlib.sha256(x).digest()[:8],"big"))

def generate_worlds(n_per_family=30,seed=20261001):
    if n_per_family<2 or n_per_family%2: raise ValueError("worlds_per_family must be even and >=2")
    rng=random.Random(seed); out=[]
    for fam in FAMILIES:
        providers=[TERMINALS[0]]*(n_per_family//2)+[TERMINALS[1]]*(n_per_family//2);rng.shuffle(providers)
        for i,p in enumerate(providers):
            if fam=="filename": vals=(f"alpha_{rng.randrange(100000,999999)}.txt",f"beta_{rng.randrange(100000,999999)}.txt")
            elif fam=="destination": vals=(f"archive_{rng.randrange(100,999)}",f"archive_{rng.randrange(100,999)}")
            else:
                seq=rng.sample([f"{c}{n}" for c in "KMPQ" for n in range(10)],3); vals=(" ".join(seq)," ".join(reversed(seq)))
            out.append(World(f"{fam}-{i:04d}",fam,*vals,p,seed))
    return out

def make_graph(world,depth,distractors,layout,seed):
    if depth not in DEPTHS or distractors not in DISTRACTORS or layout not in LAYOUTS: raise ValueError("invalid factorial cell")
    ix=int(world.world_id.rsplit("-",1)[1]); nodes=[f"Node {x}" for x in "ABCDEFGHJKLMN"]
    r=_rng(seed,world.world_id,"path");r.shuffle(nodes)
    chain=["Field F",*nodes[:depth-1],world.resolved_provider]
    relevant=[{"edge_id":f"path_{i}","source":chain[i],"target":chain[i+1],"relevant":True} for i in range(depth)]
    # Each distractor is a disconnected, unique directed edge.
    drnodes=[f"Node X{i}" for i in range(distractors*2)]
    distract=[{"edge_id":f"distractor_{i}","source":drnodes[2*i],"target":drnodes[2*i+1],"relevant":False} for i in range(distractors)]
    if layout=="compact": edges=relevant+distract
    else:
        edges=[]; slots=list(range(depth+distractors)); rel_slots=sorted(round(i*(len(slots)-1)/max(depth-1,1)) for i in range(depth))
        # Guarantee both ends and interleave as much as available.
        if depth==1: rel_slots=[len(slots)//2]
        it=iter(distract); ri=di=0
        for pos in range(len(slots)):
            if pos in rel_slots: edges.append(relevant[ri]);ri+=1
            else: edges.append(distract[di]);di+=1
    pos=[i for i,e in enumerate(edges) if e["relevant"]]
    gaps=[b-a for a,b in zip(pos,pos[1:])]
    return {"edges":edges,"relevant_path_depth":depth,"distractor_edge_count":distractors,"layout":layout,
      "relevant_edge_positions":pos,"first_relevant_position":pos[0],"last_relevant_position":pos[-1],
      "relevant_span":pos[-1]-pos[0]+1,"mean_relevant_gap":sum(gaps)/len(gaps) if gaps else 0,
      "relation_block_length":len(edges),"terminal_provider":world.resolved_provider}

def render(world,depth,k,layout,task,seed,condition="BASE"):
    g=make_graph(world,depth,k,layout,seed); rel=[e for e in g["edges"] if e["relevant"]]
    lines=[f"QUERIED ACTION FIELD: {world.task_family} (Field F)","","PROVIDER LINKS",*(f"{e['source']} -> {e['target']}" for e in g["edges"]),"",
      "VALUES",f"Source S -> {world.source_value}",f"Default policy -> {world.default_value}","",
      "RULE","Start from the queried field. Follow provider links until reaching a terminal provider. Use that provider's value.",""]
    if condition=="BRIDGE": lines += [f"Resolved provider for Field F: {world.resolved_provider}.",""]
    if condition=="NEUTRAL": lines += ["Reference note: marker R17 is recorded.",""]
    q=("Which terminal provider does Field F resolve to?" if task=="comprehension" else
       {"filename":"What filename should be used?","ordering":"What final sequence should be used?","destination":"What destination should be used?"}[world.task_family])
    lines += ["QUESTION: "+q,"Return only the exact answer text."]
    p="\n".join(lines)
    positions=[i for i,e in enumerate(g["edges"]) if e["relevant"]]
    g["relevant_edge_statement_indices"]=positions
    # token indices are populated by tokenizer audit; statement positions are always available.
    row={"world_id":world.world_id,"factorial_cell_id":f"d{depth}_k{k}_{layout}","task_family":world.task_family,
      "depth":depth,"distractor_count":k,"layout":layout,"task":task,"condition":condition,
      "world":world.to_dict(),"graph":g,"prompt":p,"correct_candidate":world.resolved_provider if task=="comprehension" else world.correct_value,
      "incorrect_candidate":(TERMINALS[1] if world.resolved_provider==TERMINALS[0] else TERMINALS[0]) if task=="comprehension" else world.alternative_value}
    return row

def generate(config,world_override=None):
    seed=int(config["seed"]); worlds=world_override if world_override is not None else generate_worlds(int(config["worlds_per_family"]),seed); rows=[]
    for w in worlds:
      for d in DEPTHS:
       for k in DISTRACTORS:
        for layout in LAYOUTS:
         for task in ("comprehension","application"): rows.append(render(w,d,k,layout,task,seed))
      # Direct semantic-action upper bound, scored only for application.
      p=f"FINAL {w.task_family}: {w.correct_value}. Use {w.correct_value}. Return only the exact answer text."
      rows.append({"world_id":w.world_id,"factorial_cell_id":"DIRECT","task_family":w.task_family,"depth":"DIRECT","distractor_count":None,"layout":"DIRECT","task":"application","condition":"DIRECT","world":w.to_dict(),"graph":None,"prompt":p,"correct_candidate":w.correct_value,"incorrect_candidate":w.alternative_value})
      for d in (3,4,5):
       for cond in ("BASE","NEUTRAL","BRIDGE"):
        if cond=="BASE": continue
        rows.append(render(w,d,4,"dispersed","application",seed,cond))
       flat=render(w,1,4,"dispersed","application",seed);flat["depth"]=d;flat["factorial_cell_id"]=f"FLAT_d{d}_k4_dispersed";flat["condition"]="FLATTENED";rows.append(flat)
    audit=validate(rows,worlds)
    return rows,audit

def validate(rows,worlds):
    primary=[r for r in rows if r["condition"]=="BASE"]; expected=len(worlds)*5*4*2*2
    assert len(primary)==expected,(len(primary),expected)
    by={(r["world_id"],r["depth"],r["distractor_count"],r["layout"],r["task"]):r for r in primary}
    assert len(by)==len(primary)
    for r in primary:
      g=r["graph"]; d=r["depth"]; k=r["distractor_count"]; es=g["edges"]
      assert sum(e["relevant"] for e in es)==d and sum(not e["relevant"] for e in es)==k
      assert g["distractor_edge_count"]==k and g["relevant_path_depth"]==d
      assert g["terminal_provider"]==r["world"]["resolved_provider"]
      relevant={x for e in es if e["relevant"] for x in (e["source"],e["target"])}
      assert all(e["source"] not in relevant and e["target"] not in relevant for e in es if not e["relevant"])
      # Exactly one route from query and no route to the other terminal.
      adj=defaultdict(list)
      for e in es: adj[e["source"]].append(e["target"])
      stack=[("Field F",())]; reached=[]
      while stack:
        n,path=stack.pop()
        if n in TERMINALS: reached.append((n,len(path)));continue
        assert n not in path
        for nxt in adj[n]:stack.append((nxt,path+(n,)))
      assert reached==[(r["world"]["resolved_provider"],d)]
      if g["layout"]=="compact": assert g["relevant_edge_positions"]==list(range(d))
    # Same world semantic invariants across all rendered factorial rows.
    for w in worlds:
      rs=[r for r in primary if r["world_id"]==w.world_id]
      assert len(rs)==5*4*2*2
      assert len({r["world"]["resolved_provider"] for r in rs})==1
      assert len({(r["world"]["source_value"],r["world"]["default_value"],r["task_family"]) for r in rs})==1
      assert len({(r["world_id"],r["depth"],r["distractor_count"],r["task"]) for r in rs})==5*4*2
      for d in DEPTHS:
       for k in DISTRACTORS:
        a=by[(w.world_id,d,k,"compact","application")];b=by[(w.world_id,d,k,"dispersed","application")]
        assert sorted((e["source"],e["target"],e["relevant"]) for e in a["graph"]["edges"])==sorted((e["source"],e["target"],e["relevant"]) for e in b["graph"]["edges"])
    balance={f:{p:sum(w.task_family==f and w.resolved_provider==p for w in worlds) for p in TERMINALS} for f in FAMILIES}
    assert all(abs(x[TERMINALS[0]]-x[TERMINALS[1]])<=1 for x in balance.values())
    return {"passed":True,"worlds":len(worlds),"primary_rows":len(primary),"total_rows":len(rows),"cells_per_world":5*4*2*2,"terminal_balance":balance,"provider_balance_exact":all(x[TERMINALS[0]]==x[TERMINALS[1]] for x in balance.values())}
