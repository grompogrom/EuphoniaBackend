import base64
import os
import pickle

import numpy as np
import onnxruntime as rt
from background_task import background
from django.utils import timezone

from generator.engine import MIDI

print(os.getcwd())

from generator.engine.midi_tokenizer import MIDITokenizer
import tqdm


tokenizer = MIDITokenizer()
# model_base_path = "model_base_touhou.onnx"
# model_token_path = "model_token_touhou.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
try:
    model_base_path = "euphonia/generator/engine/model_base_touhou.onnx"
    model_token_path = "euphonia/generator/engine/model_token_touhou.onnx"
    model_base = rt.InferenceSession(model_base_path, providers=providers)
    model_token = rt.InferenceSession(model_token_path, providers=providers)
except Exception:
    model_base_path = "generator/engine/model_base_touhou.onnx"
    model_token_path = "generator/engine/model_token_touhou.onnx"
    model_base = rt.InferenceSession(model_base_path, providers=providers)
    model_token = rt.InferenceSession(model_token_path, providers=providers)


def softmax(x, axis):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def sample_top_p_k(probs, p, k):
    probs_idx = np.argsort(-probs, axis=-1)
    probs_sort = np.take_along_axis(probs, probs_idx, -1)
    probs_sum = np.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    mask = np.zeros(probs_sort.shape[-1])
    mask[:k] = 1
    probs_sort = probs_sort * mask
    probs_sort /= np.sum(probs_sort, axis=-1, keepdims=True)
    shape = probs_sort.shape
    probs_sort_flat = probs_sort.reshape(-1, shape[-1])
    probs_idx_flat = probs_idx.reshape(-1, shape[-1])
    next_token = np.stack([np.random.choice(idxs, p=pvals) for pvals, idxs in zip(probs_sort_flat, probs_idx_flat)])
    next_token = next_token.reshape(*shape[:-1])
    return next_token


def run(mid, gen_events, path_to_save, temperature=1, top_p=0.98, top_k=12, disable_channels=None, disable_patch_change=False):
    """

    :param mid: input midi as binary
    :param gen_events: num of events to generate
    :param temperature: [0.1 - 1.2]
    :param top_p: [0.1 - 1]
    :param top_k: [1 - 20]
    :param disable_channels: name of channels that need to disable
    :param disable_patch_change:
    :return:
    """

    mid_seq = []
    mid = tokenizer.tokenize(MIDI.midi2score(mid))
    mid = np.asarray(mid, dtype=np.int64)
    max_len = len(mid) + int(gen_events)
    for token_seq in mid:
        mid_seq.append(token_seq.tolist())
    generator = generate(mid, max_len=max_len, temp=temperature, top_k=top_k, top_p=top_p,
                         disable_patch_change=disable_patch_change,
                         disable_control_change=False,
                         disable_channels=disable_channels)
    for i, token_seq in enumerate(generator):
        mid_seq.append(token_seq)
    midi = tokenizer.detokenize(mid_seq)
    with open(path_to_save, 'wb') as f:
        f.write(MIDI.score2midi(midi))


def generate(prompt=None, max_len=512, temp=1.0, top_p=0.98, top_k=20,
             disable_patch_change=False, disable_control_change=False, disable_channels=None):
    print(os.getcwd())
    model_base_path = "euphonia/generator/engine/model_base_touhou.onnx"
    model_token_path = "euphonia/generator/engine/model_token_touhou.onnx"
    model_base = rt.InferenceSession(model_base_path, providers=providers)
    model_token = rt.InferenceSession(model_token_path, providers=providers)
    if disable_channels is not None:
        disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels]
    else:
        disable_channels = []
    max_token_seq = tokenizer.max_token_seq
    if prompt is None:
        input_tensor = np.full((1, max_token_seq), tokenizer.pad_id, dtype=np.int64)
        input_tensor[0, 0] = tokenizer.bos_id  # bos
    else:
        prompt = prompt[:, :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(prompt, ((0, 0), (0, max_token_seq - prompt.shape[-1])),
                            mode="constant", constant_values=tokenizer.pad_id)
        input_tensor = prompt
    input_tensor = input_tensor[None, :, :]
    cur_len = input_tensor.shape[1]
    bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
    with bar:
        while cur_len < max_len:
            end = False
            hidden = model_base.run(None, {'x': input_tensor})[0][:, -1]
            next_token_seq = np.empty((1, 0), dtype=np.int64)
            event_name = ""
            for i in range(max_token_seq):
                mask = np.zeros(tokenizer.vocab_size, dtype=np.int64)
                if i == 0:
                    mask_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                    if disable_patch_change:
                        mask_ids.remove(tokenizer.event_ids["patch_change"])
                    if disable_control_change:
                        mask_ids.remove(tokenizer.event_ids["control_change"])
                    mask[mask_ids] = 1
                else:
                    param_name = tokenizer.events[event_name][i - 1]
                    mask_ids = tokenizer.parameter_ids[param_name]
                    if param_name == "channel":
                        mask_ids = [i for i in mask_ids if i not in disable_channels]
                    mask[mask_ids] = 1
                logits = model_token.run(None, {'x': next_token_seq, "hidden": hidden})[0][:, -1:]
                scores = softmax(logits / temp, -1) * mask
                sample = sample_top_p_k(scores, top_p, top_k)
                if i == 0:
                    next_token_seq = sample
                    eid = sample.item()
                    if eid == tokenizer.eos_id:
                        end = True
                        break
                    event_name = tokenizer.id_events[eid]
                else:
                    next_token_seq = np.concatenate([next_token_seq, sample], axis=1)
                    if len(tokenizer.events[event_name]) == i:
                        break
            if next_token_seq.shape[1] < max_token_seq:
                next_token_seq = np.pad(next_token_seq, ((0, 0), (0, max_token_seq - next_token_seq.shape[-1])),
                                        mode="constant", constant_values=tokenizer.pad_id)
            next_token_seq = next_token_seq[None, :, :]
            input_tensor = np.concatenate([input_tensor, next_token_seq], axis=1)
            cur_len += 1
            bar.update(1)
            yield next_token_seq.reshape(-1)
            if end:
                break


def decode_base64_to_binary(encoding: str) -> bytes:
    data = encoding.rsplit(",", 1)[-1]
    return base64.b64decode(data)


@background(schedule=0)
def generate_from_binary_str(bin: str, count, token: str):
    midi = base64.b64decode(bin)
    path_to_save = f"euphonia/cache/{token}.mid"
    run(midi, count, path_to_save)


if __name__ == '__main__':
    print(os.getcwd())
    input_midi_path = "../../cache/input.mid"
    with open(input_midi_path, "rb") as data:
        numpy_data = data.read()
    run(numpy_data, 10,)
