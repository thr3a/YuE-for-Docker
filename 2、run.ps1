# run script by @bdsqlsz

$profile_set = 3 #5 for highvarm, 4 for high, 3 for normal,2 for low, 1 for ultra

# Model arguments
#m-a-p/YuE-s1-7B-anneal-en-cot |  m-a-p/YuE-s1-7B-anneal-en-icl   | m-a-p/YuE-s1-7B-anneal-jp-kr-cot 
#m-a-p/YuE-s1-7B-anneal-jp-kr-icl |  m-a-p/YuE-s1-7B-anneal-zh-cot |  m-a-p/YuE-s1-7B-anneal-zh-icl 
#Alissonerdx/YuE-s1-7B-anneal-en-cot-int8 | Alissonerdx/YuE-s1-7B-anneal-zh-cot-int8  | Alissonerdx/YuE-s1-7B-anneal-jp-kr-cot 
$stage1_model = "m-a-p/YuE-s1-7B-anneal-en-cot"
$stage2_model = "m-a-p/YuE-s2-1B-general"
$max_new_tokens = 3000 #1~16384
$run_n_segments = 2
$stage2_batch_size = 4

# Prompt arguments
$genre_txt = "prompt_examples/genre.txt"  # Required
$lyrics_txt = "prompt_examples/lyrics.txt"  # Required

# ICL model need
$use_audio_prompt = $false
$audio_prompt_path = ""
$prompt_start_time = 0.0
$prompt_end_time = 30.0

# Output arguments
$output_dir = "../output"
$keep_intermediate = $true
$disable_offload_model = $false
$cuda_idx = 0

# XCodec and upsampler config
$basic_model_config = "./xcodec_mini_infer/final_ckpt/config.yaml"
$resume_path = "./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth"
$config_path = "./xcodec_mini_infer/decoders/config.yaml"
$vocal_decoder_path = "./xcodec_mini_infer/decoders/decoder_131000.pth"
$inst_decoder_path = "./xcodec_mini_infer/decoders/decoder_151000.pth"
$rescale = $true

# Activate python venv
Set-Location $PSScriptRoot

if ($env:OS -ilike "*windows*") {
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = $PSScriptRoot + "\huggingface"
$Env:TORCH_HOME = $PSScriptRoot + "\torch"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:CUDA_HOME = "${env:CUDA_PATH}"
$ext_args = [System.Collections.ArrayList]::new()

if ($profile_set -ne 3) {
  [void]$ext_args.Add("--profile=$profile_set")
}

# Add model arguments
[void]$ext_args.Add("--stage1_model=$stage1_model")
[void]$ext_args.Add("--stage2_model=$stage2_model")
[void]$ext_args.Add("--max_new_tokens=$max_new_tokens")
[void]$ext_args.Add("--run_n_segments=$run_n_segments")
[void]$ext_args.Add("--stage2_batch_size=$stage2_batch_size")

# Add prompt arguments if provided
if ($genre_txt) {
  [void]$ext_args.Add("--genre_txt=$genre_txt")
}
if ($lyrics_txt) {
  [void]$ext_args.Add("--lyrics_txt=$lyrics_txt")
}
if ($use_audio_prompt) {
  [void]$ext_args.Add("--use_audio_prompt")
  if ($audio_prompt_path) {
    [void]$ext_args.Add("--audio_prompt_path=$audio_prompt_path")
  }
  [void]$ext_args.Add("--prompt_start_time=$prompt_start_time")
  [void]$ext_args.Add("--prompt_end_time=$prompt_end_time")
}

# Add output arguments
[void]$ext_args.Add("--output_dir=$output_dir")
if ($keep_intermediate) {
  [void]$ext_args.Add("--keep_intermediate")
}
if ($disable_offload_model) {
  [void]$ext_args.Add("--disable_offload_model")
}
[void]$ext_args.Add("--cuda_idx=$cuda_idx")

# Add XCodec and upsampler config
if ($basic_model_config) {
  [void]$ext_args.Add("--basic_model_config=$basic_model_config")
}
if ($resume_path) {
  [void]$ext_args.Add("--resume_path=$resume_path")
}
if ($config_path) {
  [void]$ext_args.Add("--config_path=$config_path")
}
if ($vocal_decoder_path) {
  [void]$ext_args.Add("--vocal_decoder_path=$vocal_decoder_path")
}
if ($inst_decoder_path) {
  [void]$ext_args.Add("--inst_decoder_path=$inst_decoder_path")
}

if ($rescale) {
  [void]$ext_args.Add("--rescale")
}

Set-Location $PSScriptRoot/inference

python infer.py $ext_args

Read-Host | Out-Null ;
