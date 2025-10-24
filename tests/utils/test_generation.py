from types import SimpleNamespace

import torch

from ssm.utils import generation


class CollectingStreamer:
    def __init__(self) -> None:
        self.tokens: list[torch.Tensor] = []
        self.ended = False

    def put(self, token: torch.Tensor) -> None:
        self.tokens.append(token.clone())

    def end(self) -> None:
        self.ended = True


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self._step = 0

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return {"batch": batch_size, "max_seqlen": max_seqlen, "dtype": dtype}

    def forward(self, input_ids, inference_params=None, num_last_tokens: int = 0):
        batch, seqlen = input_ids.shape
        logits = torch.full(
            (batch, seqlen, self.vocab_size),
            fill_value=-1e3,
            device=input_ids.device,
        )
        token = self._step % self.vocab_size
        logits[:, -1, token] = 1e3
        self._step += 1
        if inference_params is not None:
            inference_params["position"] = inference_params.get("position", 0) + seqlen
        return SimpleNamespace(logits=logits)


def test_decode_teacher_forcing_with_streamer():
    model = DummyModel(vocab_size=6)
    model.train()
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    teacher = torch.tensor([[3, 4, 5]], dtype=torch.long)
    streamer = CollectingStreamer()

    output = generation.decode(
        input_ids,
        model,
        max_length=5,
        teacher_outputs=teacher,
        output_scores=True,
        enable_timing=True,
        streamer=streamer,
    )

    assert model.training is True  # training mode restored
    assert output.sequences.shape == (1, 5)
    assert torch.equal(output.sequences[0, -3:], teacher[0])
    assert output.scores is not None
    assert len(output.scores) == 3
    assert all(score.shape[-1] == 6 for score in output.scores)
    assert streamer.ended is True
    assert len(streamer.tokens) == teacher.shape[1] + 1
    assert torch.equal(streamer.tokens[0], input_ids.cpu())
    streamed = torch.stack(streamer.tokens[1:], dim=1)
    assert torch.equal(streamed, teacher)
    assert output.timings is not None
    assert output.timings["prefill"] >= 0.0
    assert output.timings["decode"] >= 0.0


def test_decode_greedy_sampling_without_scores():
    model = DummyModel(vocab_size=5)
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)

    output = generation.decode(
        input_ids,
        model,
        max_length=5,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
    )

    assert output.scores is None
    expected_suffix = torch.tensor([0, 1, 2], dtype=torch.long)
    assert torch.equal(output.sequences[0, -3:], expected_suffix)


def test_decode_truncates_prompt_only():
    model = DummyModel()
    input_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)
    streamer = CollectingStreamer()

    output = generation.decode(
        input_ids,
        model,
        max_length=2,
        streamer=streamer,
        enable_timing=True,
    )

    assert output.sequences.shape == (1, 2)
    assert torch.equal(output.sequences, input_ids[:, :2])
    assert output.scores is None
    assert streamer.ended is True
    assert output.timings == {"prefill": 0.0, "decode": 0.0}


def test_decode_stops_on_eos():
    model = DummyModel(vocab_size=6)
    input_ids = torch.tensor([[1]], dtype=torch.long)
    teacher = torch.tensor([[2, 0, 4]], dtype=torch.long)

    output = generation.decode(
        input_ids,
        model,
        max_length=4,
        teacher_outputs=teacher,
        eos_token_id=0,
    )

    assert output.sequences.shape[1] == 3  # prompt + eos
    assert output.sequences[0, -1].item() == 0
