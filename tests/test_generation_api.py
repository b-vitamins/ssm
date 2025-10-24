import torch

from ssm.models import MambaConfig, MambaLMHeadModel
from ssm.models.lm import GenerationOutput


def test_model_generate_contract_returns_sequences():
    cfg = MambaConfig(d_model=32, n_layer=2, vocab_size=64)
    model = MambaLMHeadModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 4))
    max_length = 6

    sequences = model.generate(input_ids=input_ids, max_length=max_length)

    assert isinstance(sequences, torch.Tensor)
    assert sequences.shape == (2, max_length)
    assert torch.equal(sequences[:, : input_ids.size(1)], input_ids)


def test_model_generate_return_dict_with_scores():
    cfg = MambaConfig(d_model=16, n_layer=1, vocab_size=32)
    model = MambaLMHeadModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 3))

    output = model.generate(
        input_ids=input_ids,
        max_length=5,
        top_k=4,
        temperature=0.8,
        return_dict_in_generate=True,
        output_scores=True,
    )

    assert isinstance(output, GenerationOutput)
    assert output.sequences.shape == (1, 5)
    scores = output.scores
    assert scores is not None
    assert len(scores) == 2
    assert all(score.shape[-1] == cfg.vocab_size for score in scores)


def test_model_generate_teacher_forcing_streamer():
    class CollectingStreamer:
        def __init__(self) -> None:
            self.tokens: list[torch.Tensor] = []
            self.ended = False

        def put(self, token: torch.Tensor) -> None:
            self.tokens.append(token.clone())

        def end(self) -> None:
            self.ended = True

    cfg = MambaConfig(d_model=8, n_layer=1, vocab_size=16)
    model = MambaLMHeadModel(cfg)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    teacher = torch.tensor([[3, 4, 5]], dtype=torch.long)
    streamer = CollectingStreamer()

    sequences = model.generate(
        input_ids=input_ids,
        max_length=5,
        teacher_outputs=teacher,
        streamer=streamer,
        repetition_penalty=1.1,
    )

    assert isinstance(sequences, torch.Tensor)
    expected = torch.cat([input_ids, teacher[:, :3]], dim=1)
    assert torch.equal(sequences, expected)
    assert streamer.ended
    assert len(streamer.tokens) == 4  # prompt + three generated steps
    assert torch.equal(streamer.tokens[0], input_ids.cpu())
    generated = torch.stack(streamer.tokens[1:], dim=1)
    assert torch.equal(generated, teacher)
