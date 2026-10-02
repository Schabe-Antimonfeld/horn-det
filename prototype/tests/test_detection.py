import numpy as np
import pytest

from horn.datasets import Recording, split_groups, validate_records, window_label
from horn.features import FFTGate, Gate, mel_bank, mfcc, PCEN, statistics


def test_mel_triangles_and_constant_mfcc():
    bank = mel_bank()
    assert bank.shape == (40, 513)
    assert np.all(bank >= 0) and np.all(bank <= 1)
    features = mfcc(np.full((24, 40), 2.0))
    assert features[0, 0] == pytest.approx(2 * np.sqrt(40))
    np.testing.assert_allclose(features[:, 1:], 0, atol=1e-14)
    assert statistics(features).shape == (26,)


def test_pcen_streaming_has_no_future_dependency():
    energy = np.random.default_rng(5).uniform(0, 20, size=(60, 40))
    reference = PCEN().feed(energy)
    stream = PCEN()
    np.testing.assert_array_equal(np.vstack((stream.feed(energy[:19]), stream.feed(energy[19:]))), reference)
    changed = energy.copy()
    changed[30:] *= 10
    np.testing.assert_array_equal(PCEN().feed(changed)[:30], reference[:30])


def test_fft_gate_requires_temporal_consistency():
    detector = FFTGate(settings=Gate(consecutive=2))
    tone = np.zeros(513)
    tone[15] = 1000
    assert not detector.feed(tone)[0]
    assert detector.feed(tone)[0]
    assert not detector.feed(np.zeros(513))[0]


def test_cross_device_grouping_and_interval_labels():
    records = [Recording(f"{index}.wav", f"group-{index}", "unassigned", index % 2,
                         event="same-horn" if index < 2 else str(index)) for index in range(8)]
    split = split_groups(records, seed=4)
    assert split[0].split == split[1].split
    assert len({record.split for record in split}) == 3
    with pytest.raises(ValueError):
        validate_records([Recording("a.wav", "g", "train", 0), Recording("b.wav", "g", "test", 0)])
    with pytest.raises(ValueError):
        window_label(Recording("a.wav", "g", "train", 1), 0, 0.267)
    record = Recording("a.wav", "g", "train", 1, ((0.4, 0.45),))
    assert window_label(record, 0, 0.267) == 0
    assert window_label(record, 0.3, 0.567) == 1
    assert window_label(record, 0.44, 0.7) is None


def test_crnn_chunks_match_and_future_frames_cannot_change_past():
    torch = pytest.importorskip("torch")
    from horn.models import CRNN
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = CRNN().eval()
    values = torch.randn(2, 1, 40, 24)
    with torch.no_grad():
        expected, _ = model.sequence(values)
        first, state = model.sequence(values[..., :9])
        second, _ = model.sequence(values[..., 9:], state)
        torch.testing.assert_close(torch.cat((first, second), dim=1), expected)
        changed = values.clone()
        changed[..., 9:] *= 20
        actual, _ = model.sequence(changed)
        torch.testing.assert_close(actual[:, :9], expected[:, :9])


def test_tiny_training_distillation_and_model_cost(tmp_path):
    torch = pytest.importorskip("torch")
    from horn.config import Config
    from horn.models import TinyCNN, distillation_loss
    from horn.training import train_model, model_cost, save_model
    torch.set_num_threads(2)
    torch.manual_seed(2)
    labels = np.tile([0, 1], 12)
    features = np.broadcast_to((labels * 2 - 1)[:, None, None, None], (24, 1, 40, 24)).copy().astype(np.float32)
    dataset = {"features": features, "labels": labels, "groups": np.array([str(index) for index in range(24)]),
               "splits": np.array(["train"] * 16 + ["validation"] * 4 + ["test"] * 4)}
    model = TinyCNN()
    history = train_model(model, dataset, epochs=12, batch_size=8, learning_rate=0.01)
    prediction = model(torch.as_tensor(features[-4:])).argmax(dim=-1).numpy()
    np.testing.assert_array_equal(prediction, labels[-4:])
    assert history[-1]["validation_loss"] < history[0]["validation_loss"]
    student = torch.randn(4, 2, requires_grad=True)
    teacher = torch.randn(4, 2, requires_grad=True)
    loss = distillation_loss(student, teacher, torch.tensor([0, 1, 0, 1]))
    loss.backward()
    assert student.grad is not None and teacher.grad is None
    cost = model_cost(model, torch.zeros(1, 1, 40, 24))
    assert cost["parameters"] == 1834
    assert cost["macs"] > 0
    path = tmp_path / "tiny.pt"
    save_model(model, path, "tiny", Config(), "logmel", history)
    restored = TinyCNN()
    restored.load_state_dict(torch.load(path, weights_only=True)["state"])
    torch.testing.assert_close(model(torch.as_tensor(features)), restored(torch.as_tensor(features)))


def test_svm_calibration_roundtrip_and_test_labels_are_unused(tmp_path):
    pytest.importorskip("sklearn")
    from horn.training import LinearSVM
    labels = np.tile([0, 1], 9)
    dataset = {"features": (labels * 2 - 1)[:, None].astype(float), "labels": labels.copy(),
               "groups": np.array([str(index) for index in range(18)]),
               "splits": np.array(["train"] * 10 + ["validation"] * 4 + ["test"] * 4)}
    model = LinearSVM().fit(dataset)
    before = model.probability(dataset["features"])
    assert np.all(before[labels == 1] > 0.5) and np.all(before[labels == 0] < 0.5)
    dataset["labels"][-4:] = 1 - labels[-4:]
    np.testing.assert_array_equal(LinearSVM().fit(dataset).probability(dataset["features"]), before)
    path = tmp_path / "svm.npz"
    model.save(path)
    np.testing.assert_array_equal(LinearSVM.load(path).probability(dataset["features"]), before)


def test_transfer_refuses_random_encoder():
    pytest.importorskip("torch")
    from horn.pretrained import Cnn6, TransferCNN
    with pytest.raises(ValueError):
        TransferCNN(Cnn6())


@pytest.mark.parametrize("name", ["mlp", "crnn"])
def test_candidate_training_and_distillation_schedule(name):
    torch = pytest.importorskip("torch")
    from horn.models import create_model
    from horn.training import train_model
    torch.set_num_threads(2)
    shape = (8, 26) if name == "mlp" else (8, 1, 40, 24)
    dataset = {"features": np.random.default_rng(4).normal(size=shape).astype(np.float32),
               "labels": np.tile([0, 1], 4), "groups": np.array([str(index) for index in range(8)]),
               "splits": np.array(["train"] * 4 + ["validation"] * 2 + ["test"] * 2)}
    model = create_model(name)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    history = train_model(model, dataset, epochs=1, teacher_logits=np.tile([-1., 1.], (8, 1)))
    assert np.isfinite(history[0]["train_loss"])
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_interval_aware_feature_preparation(tmp_path):
    import soundfile as sf
    from horn.datasets import prepare
    path = tmp_path / "clip.wav"
    waveform = np.zeros(48000)
    waveform[24000:28800] = np.sin(2 * np.pi * 800 * np.arange(4800) / 48000)
    sf.write(path, waveform, 48000)
    dataset = prepare([Recording(str(path), "event-a", "train", 1, ((0.5, 0.6),))], frontend="mfcc")
    assert dataset["features"].shape[1] == 26
    assert np.unique(dataset["labels"]).tolist() == [0, 1]
    assert np.all(dataset["groups"] == "event-a")
    assert len(np.unique(dataset["identities"])) == len(dataset["labels"])


def test_real_panns_weights_frontend_and_frozen_encoder():
    torch = pytest.importorskip("torch")
    from pathlib import Path
    from horn.pretrained import Cnn6, TransferCNN
    path = Path(__file__).resolve().parents[1] / "weights" / "Cnn6.pth"
    if not path.exists():
        pytest.skip("Download the verified optional Cnn6 weights to run the integration test")
    torch.set_num_threads(2)
    encoder = Cnn6().load_pretrained(path).eval()
    waveform = np.sin(2 * np.pi * 440 * np.arange(16000) / 32000)
    feature = encoder.frontend(waveform, 32000)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)["model"]
    padded = torch.nn.functional.pad(torch.tensor(waveform, dtype=torch.float32)[None, None], (512, 512), mode="reflect")
    real = torch.nn.functional.conv1d(padded, checkpoint["spectrogram_extractor.stft.conv_real.weight"], stride=320)
    imaginary = torch.nn.functional.conv1d(padded, checkpoint["spectrogram_extractor.stft.conv_imag.weight"], stride=320)
    reference = 10 * torch.log10(torch.clamp((real.square() + imaginary.square()).transpose(1, 2)
                                            @ checkpoint["logmel_extractor.melW"], min=1e-10))
    torch.testing.assert_close(feature[0], reference, rtol=0.005, atol=0.4)
    transfer = TransferCNN(encoder).train()
    assert not encoder.training and not encoder.bn0.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert transfer(feature).shape == (1, 2)
    transfer = TransferCNN(encoder, fine_tune=True).train()
    assert encoder.conv_block4.conv1.weight.requires_grad
    assert not encoder.conv_block1.conv1.weight.requires_grad
