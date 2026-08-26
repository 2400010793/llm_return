from scripts.build_three_model_four_prompt_fair import target_indices
from scripts.run_three_model_prompt_fusion import groups


def test_target_indices_cover_exact_phrase_across_tokenizers():
    assert target_indices(["分", "析", "股", "票", "超", "额", "收", "益"], "超额收益") == [4, 5, 6, 7]
    assert target_indices(["▁", "分析", "股票", "超", "额", "收益"], "超额收益") == [3, 4, 5]
    assert target_indices(["分析", "股票", "超额", "收益", "。"], "超额收益") == [2, 3]


def test_fusion_groups_are_complete_and_non_overlapping_where_expected():
    names = [
        f"{model}__{prompt}__{representation}"
        for model in ("roberta", "bge_m3", "qwen3_embedding_8b")
        for prompt in ("profit", "return", "excess_return", "loss")
        for representation in ("token", "body")
    ]
    result = groups(names)
    assert len(result["all_12_tokens"]) == 12
    assert len(result["all_12_bodies"]) == 12
    assert len(result["all_24_token_body"]) == 24
    assert len(result["model_qwen3_embedding_8b_tokens"]) == 4
    assert len(result["prompt_loss_models"]) == 3
