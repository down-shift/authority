"""Joint-tokenization semantic scoring and exact tokenization audit."""
from authority_leakage.models.continuation import continuation_encoding
from authority_leakage.schemas import Message
from authority_leakage.progress import tqdm
from .validation import summarize_tokenization


def audit_tokenization(rows,adapter):
    token_rows=[]
    for row in tqdm(rows,total=len(rows),desc="Auditing v2 token lengths",unit="prompt",leave=False):
        rendered=adapter.render([Message(role="user",content=row["prompt"])])
        context_count=len(adapter.tokenizer(rendered,add_special_tokens=False)["input_ids"])
        for candidate in (row["correct_candidate"],row["incorrect_candidate"]):
            enc=continuation_encoding(adapter.tokenizer,rendered,candidate)
            token_rows.append({"world_id":row["world_id"],"task_family":row["task_family"],
                "indirection_depth":row["indirection_depth"],"template_id":row["template_id"],
                "model":row["model"],"measurement_type":row["measurement_type"],
                "condition_type":row["condition_type"],"target_depth":row["target_depth"],
                "relevant_path_depth":row["relevant_path_depth"],"candidate":candidate,
                "prompt_token_count":context_count,"continuation_token_count":enc["token_count"],
                "boundary_overlap":enc["boundary_overlap"],
                "prompt_prefix_retokenized":enc["prompt_prefix_retokenized"],
                "boundary_mode":enc["boundary_mode"]})
    return token_rows,summarize_tokenization(rows,token_rows)


def score_example(row,adapter,model):
    from indirection.scoring import score_example as semantic_score
    result=semantic_score(row,adapter,model)
    correct=result["candidate_scores"][row["correct_candidate"]]
    result["prompt_token_count"]=correct["prompt_token_count"]
    result["continuation_token_counts"]={c:s["token_count"] for c,s in result["candidate_scores"].items()}
    return result
