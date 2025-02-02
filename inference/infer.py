import os
import sys
from pathlib import Path
from mmgp import offload

# Add paths for xcodec modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "xcodec_mini_infer"))
sys.path.append(os.path.join(current_dir, "xcodec_mini_infer", "descriptaudiocodec"))
sys.path.append(current_dir)  # Add current directory to path

import re
import random
import uuid
import copy
from tqdm import tqdm
from collections import Counter
import argparse
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
from einops import rearrange
from transformers import (
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    BitsAndBytesConfig,
)
from omegaconf import OmegaConf
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
import copy
from collections import Counter
from models.soundstream_hubert_new import SoundStream
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched


def create_args(
    genre_txt: str = (Path(current_dir) / "prompt_examples/genre.txt").as_posix(),
    lyrics_txt: str = (Path(current_dir) / "prompt_examples/lyrics.txt").as_posix(),
    stage1_model: str = "m-a-p/YuE-s1-7B-anneal-en-cot",
    stage2_model="m-a-p/YuE-s2-1B-general",
    max_new_tokens: int = 3000,
    run_n_segments: int = 2,
    stage2_batch_size: int = 2,
    use_audio_prompt: bool = False,
    audio_prompt_path: str = "",
    prompt_start_time: float = 0.0,
    prompt_end_time: float = 30.0,
    use_dual_tracks_prompt: bool = False,
    vocal_track_prompt_path: str = "",
    instrumental_track_prompt_path: str = "",
    output_dir: str = "./output",
    keep_intermediate: bool = False,
    disable_offload_model: bool = True,
    cuda_idx: int = 0,
    seed: int = 42,
    basic_model_config: str = (
        Path(current_dir) / "xcodec_mini_infer/final_ckpt/config.yaml"
    ).as_posix(),
    resume_path: str = (
        Path(current_dir) / "xcodec_mini_infer/final_ckpt/ckpt_00360000.pth"
    ).as_posix(),
    config_path: str = (
        Path(current_dir) / "xcodec_mini_infer/decoders/config.yaml"
    ).as_posix(),
    vocal_decoder_path: str = (
        Path(current_dir) / "xcodec_mini_infer/decoders/decoder_131000.pth"
    ).as_posix(),
    inst_decoder_path: str = (
        Path(current_dir) / "xcodec_mini_infer/decoders/decoder_151000.pth"
    ).as_posix(),
    rescale: bool = False,
    compile: bool = True,
    profile: int = 3,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Model Configuration:
    parser.add_argument(
        "--stage1_model",
        type=str,
        default="m-a-p/YuE-s1-7B-anneal-en-cot",
        help="The model checkpoint path or identifier for the Stage 1 model.",
    )
    parser.add_argument(
        "--stage2_model",
        type=str,
        default="m-a-p/YuE-s2-1B-general",
        help="The model checkpoint path or identifier for the Stage 2 model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3000,
        help="The maximum number of new tokens to generate in one pass during text generation.",
    )
    parser.add_argument(
        "--run_n_segments",
        type=int,
        default=2,
        help="The number of segments to process during the generation.",
    )
    parser.add_argument(
        "--stage2_batch_size",
        type=int,
        default=4,
        help="The batch size used in Stage 2 inference.",
    )
    # Prompt
    parser.add_argument(
        "--genre_txt",
        type=str,
        required=True,
        help="The file path to a text file containing genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.",
    )
    parser.add_argument(
        "--lyrics_txt",
        type=str,
        required=True,
        help="The file path to a text file containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.",
    )
    parser.add_argument(
        "--use_audio_prompt",
        action="store_true",
        help="If set, the model will use an audio file as a prompt during generation. The audio file should be specified using --audio_prompt_path.",
    )
    parser.add_argument(
        "--audio_prompt_path",
        type=str,
        default="",
        help="The file path to an audio file to use as a reference prompt when --use_audio_prompt is enabled.",
    )
    parser.add_argument(
        "--prompt_start_time",
        type=float,
        default=0.0,
        help="The start time in seconds to extract the audio prompt from the given audio file.",
    )
    parser.add_argument(
        "--prompt_end_time",
        type=float,
        default=30.0,
        help="The end time in seconds to extract the audio prompt from the given audio file.",
    )
    parser.add_argument(
        "--use_dual_tracks_prompt",
        action="store_true",
        help="If set, the model will use dual tracks as a prompt during generation. The vocal and instrumental files should be specified using --vocal_track_prompt_path and --instrumental_track_prompt_path.",
    )
    parser.add_argument(
        "--vocal_track_prompt_path",
        type=str,
        default="",
        help="The file path to a vocal track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.",
    )
    parser.add_argument(
        "--instrumental_track_prompt_path",
        type=str,
        default="",
        help="The file path to an instrumental track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.",
    )
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The directory where generated outputs will be saved.",
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="If set, intermediate outputs will be saved during processing.",
    )
    parser.add_argument(
        "--disable_offload_model",
        action="store_true",
        help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.",
    )
    parser.add_argument("--cuda_idx", type=int, default=0)
    parser.add_argument(
        "--seed", type=int, default=42, help="An integer value to reproduce generation."
    )
    # Config for xcodec and upsampler
    parser.add_argument(
        "--basic_model_config",
        default=(
            Path(current_dir) / "xcodec_mini_infer" / "final_ckpt" / "config.yaml"
        ).as_posix(),
        help="YAML files for xcodec configurations.",
    )
    parser.add_argument(
        "--resume_path",
        default=(
            Path(current_dir) / "xcodec_mini_infer" / "final_ckpt" / "ckpt_00360000.pth"
        ).as_posix(),
        help="Path to the xcodec checkpoint.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=(
            Path(current_dir) / "xcodec_mini_infer" / "decoders" / "config.yaml"
        ).as_posix(),
        help="Path to Vocos config file.",
    )
    parser.add_argument(
        "--vocal_decoder_path",
        type=str,
        default=(
            Path(current_dir) / "xcodec_mini_infer" / "decoders" / "decoder_131000.pth"
        ).as_posix(),
        help="Path to Vocos decoder weights.",
    )
    parser.add_argument(
        "--inst_decoder_path",
        type=str,
        default=(
            Path(current_dir) / "xcodec_mini_infer" / "decoders" / "decoder_151000.pth"
        ).as_posix(),
        help="Path to Vocos decoder weights.",
    )
    parser.add_argument(
        "-r", "--rescale", action="store_true", help="Rescale output to avoid clipping."
    )
    parser.add_argument("--profile", type=int, default=3)
    parser.add_argument("--compile", action="store_true", help="Compile model.")

    args = parser.parse_args(
        [
            "--genre_txt",
            genre_txt,
            "--lyrics_txt",
            lyrics_txt,
            "--stage1_model",
            stage1_model,
            "--stage2_model",
            stage2_model,
            "--max_new_tokens",
            str(max_new_tokens),
            "--stage2_batch_size",
            str(stage2_batch_size),
            "--run_n_segments",
            str(run_n_segments),
            "--output_dir",
            output_dir,
            "--cuda_idx",
            str(cuda_idx),
            "--basic_model_config",
            basic_model_config,
            "--resume_path",
            resume_path,
            "--config_path",
            config_path,
            "--vocal_decoder_path",
            vocal_decoder_path,
            "--inst_decoder_path",
            inst_decoder_path,
            "--seed",
            str(seed),
        ]
    )
    if use_audio_prompt:
        args.use_audio_prompt = True
        args.audio_prompt_path = audio_prompt_path
        args.prompt_start_time = prompt_start_time
        args.prompt_end_time = prompt_end_time
        if use_dual_tracks_prompt:
            args.use_dual_tracks_prompt = True
            args.vocal_track_prompt_path = vocal_track_prompt_path
            args.instrumental_track_prompt_path = instrumental_track_prompt_path

    args.keep_intermediate = keep_intermediate

    args.disable_offload_model = disable_offload_model

    args.rescale = rescale

    args.profile = profile

    args.compile = compile

    return args, parser


def main(args):
    stage1_model = args.stage1_model
    stage2_model = args.stage2_model
    cuda_idx = args.cuda_idx
    max_new_tokens = args.max_new_tokens
    stage1_output_dir = os.path.join(args.output_dir, f"stage1")
    stage2_output_dir = stage1_output_dir.replace("stage1", "stage2")
    os.makedirs(stage1_output_dir, exist_ok=True)
    os.makedirs(stage2_output_dir, exist_ok=True)

    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(args.seed)

    # load tokenizer and model
    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
    print(device)
    mmtokenizer = _MMSentencePieceTokenizer(
        (Path(current_dir) / "mm_tokenizer_v0.2_hf" / "tokenizer.model").as_posix()
    )

    def load_model(model_path, quantization):
        if quantization == "bf16":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",  # To enable flashattn, you have to install flash-attn
            )
            model.to("cpu")
        elif quantization == "int8":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True  # Enable 8-bit quantization
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2",
            )
        elif quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True  # Enable 4-bit quantization
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2",
            )
        return model

    model = load_model(
        stage1_model, "int8" if stage1_model.endswith("int8") else "bf16"
    )

    # to device, if gpu is available
    model.eval()

    # if torch.__version__ >= "2.0.0":
    #     model_stage2 = torch.compile(model)

    model_stage2 = AutoModelForCausalLM.from_pretrained(
        stage2_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # device_map="auto",
    )
    model_stage2.to("cpu")
    model_stage2.eval()

    pipe = {"transformer": model, "stage2": model_stage2}

    quantizeTransformer = args.profile == 3 or args.profile == 4 or args.profile == 5

    codectool = CodecManipulator("xcodec", 0, 1)
    codectool_stage2 = CodecManipulator("xcodec", 0, 8)
    model_config = OmegaConf.load(args.basic_model_config)
    codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(
        device
    )
    parameter_dict = torch.load(
        args.resume_path, map_location="cpu", weights_only=False
    )
    codec_model.load_state_dict(parameter_dict["codec_model"])
    del parameter_dict
    codec_model.to(device)
    codec_model.eval()

    print("profile:"+ str(args.profile))

    offload.profile(
        pipe,
        profile_no=args.profile,
        quantizeTransformer=quantizeTransformer,
        compile=False,
        verboseLevel=1,
    )

    class BlockTokenRangeProcessor(LogitsProcessor):
        def __init__(self, start_id, end_id):
            self.blocked_token_ids = list(range(start_id, end_id))

        def __call__(self, input_ids, scores):
            scores[:, self.blocked_token_ids] = -float("inf")
            return scores

    def load_audio_mono(filepath, sampling_rate=16000):
        audio, sr = torchaudio.load(filepath)
        # Convert to mono
        audio = torch.mean(audio, dim=0, keepdim=True)
        # Resample if needed
        if sr != sampling_rate:
            resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
            audio = resampler(audio)
        return audio

    def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
        if len(audio_prompt.shape) < 3:
            audio_prompt.unsqueeze_(0)
        with torch.no_grad():
            raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
        raw_codes = raw_codes.transpose(0, 1)
        raw_codes = raw_codes.cpu().numpy().astype(np.int16)
        return raw_codes

    def split_lyrics(lyrics):
        pattern = r"\[(\w+)\](.*?)\n(?=\[|\Z)"
        segments = re.findall(pattern, lyrics, re.DOTALL)
        structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
        return structured_lyrics

    # Call the function and print the result
    stage1_output_set = []
    # Tips:
    # genre tags support instrumental，genre，mood，vocal timbr and vocal gender
    # all kinds of tags are needed
    if args.genre_txt.endswith(".txt"):
        with open(args.genre_txt, "r", encoding="utf-8") as f:
            genres = f.read().strip()
    else:
        genres = args.genre_txt

    if args.lyrics_txt.endswith(".txt"):
        with open(args.lyrics_txt, "r", encoding="utf-8") as f:
            lyrics = split_lyrics(f.read())
    else:
        lyrics = split_lyrics(args.lyrics_txt)

    # intruction
    full_lyrics = "\n".join(lyrics)
    prompt_texts = [
        f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"
    ]
    prompt_texts += lyrics

    random_id = uuid.uuid4()
    output_seq = None
    # Here is suggested decoding config
    top_p = 0.93
    temperature = 1.0
    repetition_penalty = 1.2
    # special tokens
    start_of_segment = mmtokenizer.tokenize("[start_of_segment]")
    end_of_segment = mmtokenizer.tokenize("[end_of_segment]")
    # Format text prompt
    run_n_segments = min(args.run_n_segments + 1, len(lyrics))
    raw_output = None
    for i, p in enumerate(tqdm(prompt_texts[:run_n_segments])):
        section_text = p.replace("[start_of_segment]", "").replace(
            "[end_of_segment]", ""
        )
        guidance_scale = 1.5 if i <= 1 else 1.2
        if i == 0:
            continue
        if i == 1:
            if args.use_dual_tracks_prompt or args.use_audio_prompt:
                if args.use_dual_tracks_prompt:
                    vocals_ids = load_audio_mono(args.vocal_track_prompt_path)
                    instrumental_ids = load_audio_mono(
                        args.instrumental_track_prompt_path
                    )
                    vocals_ids = encode_audio(
                        codec_model, vocals_ids, device, target_bw=0.5
                    )
                    instrumental_ids = encode_audio(
                        codec_model, instrumental_ids, device, target_bw=0.5
                    )
                    vocals_ids = codectool.npy2ids(vocals_ids[0])
                    instrumental_ids = codectool.npy2ids(instrumental_ids[0])
                    ids_segment_interleaved = rearrange(
                        [np.array(vocals_ids), np.array(instrumental_ids)],
                        "b n -> (n b)",
                    )
                    audio_prompt_codec = ids_segment_interleaved[
                        int(args.prompt_start_time * 50 * 2) : int(
                            args.prompt_end_time * 50 * 2
                        )
                    ]
                    audio_prompt_codec = audio_prompt_codec.tolist()
                elif args.use_audio_prompt:
                    audio_prompt = load_audio_mono(args.audio_prompt_path)
                    raw_codes = encode_audio(
                        codec_model, audio_prompt, device, target_bw=0.5
                    )
                    # Format audio prompt
                    code_ids = codectool.npy2ids(raw_codes[0])
                    audio_prompt_codec = code_ids[
                        int(args.prompt_start_time * 50) : int(
                            args.prompt_end_time * 50
                        )
                    ]  # 50 is tps of xcodec
                audio_prompt_codec_ids = (
                    [mmtokenizer.soa]
                    + codectool.sep_ids
                    + audio_prompt_codec
                    + [mmtokenizer.eoa]
                )
                sentence_ids = (
                    mmtokenizer.tokenize("[start_of_reference]")
                    + audio_prompt_codec_ids
                    + mmtokenizer.tokenize("[end_of_reference]")
                )
                head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
            else:
                head_id = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = (
                head_id
                + start_of_segment
                + mmtokenizer.tokenize(section_text)
                + [mmtokenizer.soa]
                + codectool.sep_ids
            )
        else:
            prompt_ids = (
                end_of_segment
                + start_of_segment
                + mmtokenizer.tokenize(section_text)
                + [mmtokenizer.soa]
                + codectool.sep_ids
            )

        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
        # Use window slicing in case output sequence exceeds the context of model
        max_context = 16384 - max_new_tokens - 1
        if input_ids.shape[-1] > max_context:
            print(
                f"Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens."
            )
            input_ids = input_ids[:, -(max_context):]
        with torch.no_grad():
            output_seq = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList(
                    [
                        BlockTokenRangeProcessor(0, 32002),
                        BlockTokenRangeProcessor(32016, 32016),
                    ]
                ),
                guidance_scale=guidance_scale,
            )
            if output_seq[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
            if i > 1 or raw_output is not None:
                raw_output = torch.cat(
                    [raw_output, prompt_ids, output_seq[:, input_ids.shape[-1] :]], dim=1
                )
            else:
                raw_output = output_seq

    # save raw output and check sanity
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    if len(soa_idx) != len(eoa_idx):
        raise ValueError(
            f"invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}"
        )

    vocals = []
    instrumentals = []
    range_begin = 1 if args.use_audio_prompt or args.use_dual_tracks_prompt else 0
    for i in range(range_begin, len(soa_idx)):
        codec_ids = ids[soa_idx[i] + 1 : eoa_idx[i]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
        vocals.append(vocals_ids)
        instrumentals_ids = codectool.ids2npy(
            rearrange(codec_ids, "(n b) -> b n", b=2)[1]
        )
        instrumentals.append(instrumentals_ids)
    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)
    vocal_save_path = os.path.join(
        stage1_output_dir,
        f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_vtrack".replace(
            ".", "@"
        )
        + ".npy",
    )
    inst_save_path = os.path.join(
        stage1_output_dir,
        f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_itrack".replace(
            ".", "@"
        )
        + ".npy",
    )
    np.save(vocal_save_path, vocals)
    np.save(inst_save_path, instrumentals)
    stage1_output_set.append(vocal_save_path)
    stage1_output_set.append(inst_save_path)

    # offload model
    # if not args.disable_offload_model:
    #     model.cpu()
    #     del model
    #     torch.cuda.empty_cache()

    print("Stage 2 inference...")
    # model_stage2 = AutoModelForCausalLM.from_pretrained(
    #     stage2_model,
    #     torch_dtype=torch.float16,
    #     attn_implementation="flash_attention_2"
    #     )
    # model_stage2.to(device)
    # model_stage2.eval()

    # if torch.__version__ >= "2.0.0":
    #     model_stage2 = torch.compile(model_stage2)

    def stage2_generate(model, prompt, batch_size=16):
        codec_ids = codectool.unflatten(prompt, n_quantizer=1)
        codec_ids = codectool.offset_tok_ids(
            codec_ids,
            global_offset=codectool.global_offset,
            codebook_size=codectool.codebook_size,
            num_codebooks=codectool.num_codebooks,
        ).astype(np.int32)

        # Prepare prompt_ids based on batch size or single input
        if batch_size > 1:
            codec_list = []
            for i in range(batch_size):
                idx_begin = i * 300
                idx_end = (i + 1) * 300
                codec_list.append(codec_ids[:, idx_begin:idx_end])

            codec_ids = np.concatenate(codec_list, axis=0)
            prompt_ids = np.concatenate(
                [
                    np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                    codec_ids,
                    np.tile([mmtokenizer.stage_2], (batch_size, 1)),
                ],
                axis=1,
            )
        else:
            prompt_ids = np.concatenate(
                [
                    np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                    codec_ids.flatten(),  # Flatten the 2D array to 1D
                    np.array([mmtokenizer.stage_2]),
                ]
            ).astype(np.int32)
            prompt_ids = prompt_ids[np.newaxis, ...]

        codec_ids = torch.as_tensor(codec_ids).to(device)
        prompt_ids = torch.as_tensor(prompt_ids).to(device)
        len_prompt = prompt_ids.shape[-1]

        block_list = LogitsProcessorList(
            [
                BlockTokenRangeProcessor(0, 46358),
                BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size),
            ]
        )

        # Teacher forcing generate loop
        for frames_idx in range(codec_ids.shape[1]):
            cb0 = codec_ids[:, frames_idx : frames_idx + 1]
            prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
            input_ids = prompt_ids

            with torch.no_grad():
                stage2_output = model.generate(
                    input_ids=input_ids,
                    min_new_tokens=7,
                    max_new_tokens=7,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                    logits_processor=block_list,
                )

            assert (
                stage2_output.shape[1] - prompt_ids.shape[1] == 7
            ), f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
            prompt_ids = stage2_output

        # Return output based on batch size
        if batch_size > 1:
            output = prompt_ids.cpu().numpy()[:, len_prompt:]
            output_list = [output[i] for i in range(batch_size)]
            output = np.concatenate(output_list, axis=0)
        else:
            output = prompt_ids[0].cpu().numpy()[len_prompt:]

        return output

    def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
        stage2_result = []
        for i in tqdm(range(len(stage1_output_set))):
            output_filename = os.path.join(
                stage2_output_dir, os.path.basename(stage1_output_set[i])
            )

            if os.path.exists(output_filename):
                print(f"{output_filename} stage2 has done.")
                continue

            # Load the prompt
            prompt = np.load(stage1_output_set[i]).astype(np.int32)

            # Only accept 6s segments
            output_duration = prompt.shape[-1] // 50 // 6 * 6
            num_batch = output_duration // 6

            if num_batch <= batch_size:
                # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
                output = stage2_generate(
                    model, prompt[:, : output_duration * 50], batch_size=num_batch
                )
            else:
                # If num_batch is greater than batch_size, process in chunks of batch_size
                segments = []
                num_segments = (num_batch // batch_size) + (
                    1 if num_batch % batch_size != 0 else 0
                )

                for seg in range(num_segments):
                    start_idx = seg * batch_size * 300
                    # Ensure the end_idx does not exceed the available length
                    end_idx = min(
                        (seg + 1) * batch_size * 300, output_duration * 50
                    )  # Adjust the last segment
                    current_batch_size = (
                        batch_size
                        if seg != num_segments - 1 or num_batch % batch_size == 0
                        else num_batch % batch_size
                    )
                    segment = stage2_generate(
                        model,
                        prompt[:, start_idx:end_idx],
                        batch_size=current_batch_size,
                    )
                    segments.append(segment)

                # Concatenate all the segments
                output = np.concatenate(segments, axis=0)

            # Process the ending part of the prompt
            if output_duration * 50 != prompt.shape[-1]:
                ending = stage2_generate(
                    model, prompt[:, output_duration * 50 :], batch_size=1
                )
                output = np.concatenate([output, ending], axis=0)
            output = codectool_stage2.ids2npy(output)

            # Fix invalid codes (a dirty solution, which may harm the quality of audio)
            # We are trying to find better one
            fixed_output = copy.deepcopy(output)
            for i, line in enumerate(output):
                for j, element in enumerate(line):
                    if element < 0 or element > 1023:
                        counter = Counter(line)
                        most_frequant = sorted(
                            counter.items(), key=lambda x: x[1], reverse=True
                        )[0][0]
                        fixed_output[i, j] = most_frequant
            # save output
            np.save(output_filename, fixed_output)
            stage2_result.append(output_filename)
        return stage2_result

    stage2_result = stage2_inference(
        model_stage2,
        stage1_output_set,
        stage2_output_dir,
        batch_size=args.stage2_batch_size,
    )
    print(stage2_result)
    print("Stage 2 DONE.\n")

    # convert audio tokens to audio
    def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
        folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        limit = 0.99
        max_val = wav.abs().max()
        wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
        torchaudio.save(
            str(path),
            wav,
            sample_rate=sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )

    # reconstruct tracks
    recons_output_dir = os.path.join(args.output_dir, "recons")
    recons_mix_dir = os.path.join(recons_output_dir, "mix")
    os.makedirs(recons_mix_dir, exist_ok=True)
    tracks = []
    for npy in stage2_result:
        codec_result = np.load(npy)
        decodec_rlt = []
        with torch.no_grad():
            decoded_waveform = codec_model.decode(
                torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long)
                .unsqueeze(0)
                .permute(1, 0, 2)
                .to(device)
            )
        decoded_waveform = decoded_waveform.cpu().squeeze(0)
        decodec_rlt.append(torch.as_tensor(decoded_waveform, device="cpu"))
        decodec_rlt = torch.cat(decodec_rlt, dim=-1)
        save_path = os.path.join(
            recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3"
        )
        tracks.append(save_path)
        save_audio(decodec_rlt, save_path, 16000)
    # mix tracks
    for inst_path in tracks:
        try:
            if (
                inst_path.endswith(".wav") or inst_path.endswith(".mp3")
            ) and "_itrack" in inst_path:
                # find pair
                vocal_path = inst_path.replace("_itrack", "_vtrack")
                if not os.path.exists(vocal_path):
                    continue
                # mix
                recons_mix = os.path.join(
                    recons_mix_dir,
                    os.path.basename(inst_path).replace("_itrack", "_mixed"),
                )
                vocal_stem, sr = sf.read(inst_path)
                instrumental_stem, _ = sf.read(vocal_path)
                mix_stem = (vocal_stem + instrumental_stem) / 1
                sf.write(recons_mix, mix_stem, sr)
        except Exception as e:
            print(e)

    # vocoder to upsample audios
    vocal_decoder, inst_decoder = build_codec_model(
        args.config_path, args.vocal_decoder_path, args.inst_decoder_path
    )
    vocoder_output_dir = os.path.join(args.output_dir, "vocoder")
    vocoder_stems_dir = os.path.join(vocoder_output_dir, "stems")
    vocoder_mix_dir = os.path.join(vocoder_output_dir, "mix")
    os.makedirs(vocoder_mix_dir, exist_ok=True)
    os.makedirs(vocoder_stems_dir, exist_ok=True)
    for npy in stage2_result:
        if "_itrack" in npy:
            # Process instrumental
            instrumental_output = process_audio(
                npy,
                os.path.join(vocoder_stems_dir, "itrack.mp3"),
                args.rescale,
                args,
                inst_decoder,
                codec_model,
            )
        else:
            # Process vocal
            vocal_output = process_audio(
                npy,
                os.path.join(vocoder_stems_dir, "vtrack.mp3"),
                args.rescale,
                args,
                vocal_decoder,
                codec_model,
            )
    # mix tracks
    try:
        mix_output = instrumental_output + vocal_output
        vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
        save_audio(mix_output, vocoder_mix, 44100, args.rescale)
        print(f"Created mix: {vocoder_mix}")
    except RuntimeError as e:
        print(e)
        print(
            f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}"
        )

    # Post process
    replace_low_freq_with_energy_matched(
        a_file=recons_mix,  # 16kHz
        b_file=vocoder_mix,  # 48kHz
        c_file=os.path.join(args.output_dir, os.path.basename(recons_mix)),
        cutoff_freq=5500.0,
    )

    output_audio = os.path.join(args.output_dir, os.path.basename(recons_mix))

    return output_audio


if __name__ == "__main__":

    _, parser = create_args()

    args = parser.parse_args()

    if args.use_audio_prompt and not args.audio_prompt_path:
        raise FileNotFoundError(
            "Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!"
        )

    main(args)
