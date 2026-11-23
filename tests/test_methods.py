import pytest

torch = pytest.importorskip("torch")

from vlmeval.data.scoring import is_correct, normalize  # noqa: E402
from vlmeval.methods import METHODS, Context  # noqa: E402


@pytest.fixture
def feats():
    return torch.randn(576, 8)


@pytest.mark.parametrize("keep", [288, 144, 58])
def test_token_counts(feats, keep):
    scores = torch.rand(576)
    for name in ("random", "cls_attn"):
        out = METHODS[name](feats, Context(keep=keep, cls_scores=scores, generator=torch.Generator().manual_seed(0)))
        assert out.shape == (keep, 8)
    out = METHODS["avgpool"](feats, Context(keep=keep))
    assert abs(out.shape[0] - keep) <= 0.05 * keep + 2  # h x w grid approximates keep


def test_cls_attn_keeps_top_scores(feats):
    scores = torch.zeros(576)
    scores[[3, 10, 100]] = 1.0
    out = METHODS["cls_attn"](feats, Context(keep=3, cls_scores=scores))
    assert torch.equal(out, feats[[3, 10, 100]])


def test_random_is_seeded(feats):
    mk = lambda: METHODS["random"](feats, Context(keep=50, generator=torch.Generator().manual_seed(1)))
    assert torch.equal(mk(), mk())


def test_scoring():
    assert normalize("The  Dog.") == "dog"
    assert is_correct("Yes, there is.", ["yes"], "yesno") == 1
    assert is_correct("No", ["yes"], "yesno") == 0
    assert is_correct("Coca-Cola.", ["coca-cola"], "open") == 1
    assert is_correct("coke", ["coca-cola"], "open") == 0
