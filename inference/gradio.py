import gradio as gr
import os
from inference.infer import create_args, main
from pathlib import Path
import torch


def generate_music(
    genre_txt,
    lyrics_txt,
    stage1_model,
    stage2_model,
    max_new_tokens,
    run_n_segments,
    stage2_batch_size,
    use_audio_prompt,
    audio_prompt_file,
    prompt_start_time,
    prompt_end_time,
    output_dir,
    keep_intermediate,
    disable_offload_model,
    cuda_idx,
    rescale,
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_dir = Path(output_dir).absolute().as_posix()

    # Handle audio prompt path
    audio_prompt_path = ""
    if use_audio_prompt and audio_prompt_file is not None:
        audio_prompt_path = audio_prompt_file.name

    # Create arguments using the create_args function
    args, _ = create_args(
        genre_txt=genre_txt,
        lyrics_txt=lyrics_txt,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        max_new_tokens=max_new_tokens,
        run_n_segments=run_n_segments,
        stage2_batch_size=stage2_batch_size,
        use_audio_prompt=use_audio_prompt,
        audio_prompt_path=audio_prompt_path,
        prompt_start_time=prompt_start_time,
        prompt_end_time=prompt_end_time,
        output_dir=output_dir,
        keep_intermediate=keep_intermediate,
        disable_offload_model=disable_offload_model,
        rescale=rescale,
        cuda_idx=int(cuda_idx),
    )


    output_audio = main(args)

    # Return the generated audio files
    return output_audio


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# YuE Music Generation Interface")

    with gr.Row():
        with gr.Column():
            genre_txt = gr.Textbox(
                label="Genre Tags",
                placeholder="Enter genre tags here",
            )
            lyrics_txt = gr.Textbox(
                label="Lyrics",
                placeholder="Enter lyrics here",
                lines=10,
            )
            stage1_model = gr.Dropdown(
                label="Stage 1 Model",
                choices=[
                    "m-a-p/YuE-s1-7B-anneal-en-cot",
                    "m-a-p/YuE-s1-7B-anneal-en-icl",
                    "m-a-p/YuE-s1-7B-anneal-jp-kr-cot",
                    "m-a-p/YuE-s1-7B-anneal-jp-kr-icl",
                    "m-a-p/YuE-s1-7B-anneal-zh-cot",
                    "m-a-p/YuE-s1-7B-anneal-zh-icl",
                    "Alissonerdx/YuE-s1-7B-anneal-en-cot-int8",
                    "Alissonerdx/YuE-s1-7B-anneal-en-icl-int8",
                    "Alissonerdx/YuE-s1-7B-anneal-zh-cot-int8",
                    "Alissonerdx/YuE-s1-7B-anneal-zh-icl-int8",
                    "Alissonerdx/YuE-s1-7B-anneal-jp-kr-cot-int8",
                    "Alissonerdx/YuE-s1-7B-anneal-jp-kr-icl-int8",
                ],
                value="m-a-p/YuE-s1-7B-anneal-en-cot",
            )
            stage2_model = gr.Dropdown(
                label="Stage 2 Model",
                choices=[
                    "m-a-p/YuE-s2-1B-general",
                ],
                value="m-a-p/YuE-s2-1B-general",
            )
            max_new_tokens = gr.Slider(
                label="Max New Tokens", minimum=1, maximum=16384, value=3000, step=100
            )
            run_n_segments = gr.Slider(
                label="Number of Segments", minimum=1, maximum=5, value=2, step=1
            )
            with gr.Row():
                stage2_batch_size = gr.Number(
                    label="Stage 2 Batch Size", value=4, precision=0
                )
                keep_intermediate = gr.Checkbox(
                    label="Keep Intermediate Files", value=True
                )
                disable_offload_model = gr.Checkbox(
                    label="Disable Offload Model", value=True
                )
                rescale = gr.Checkbox(label="Rescale Audio", value=False)
                cuda_idx = gr.Radio(
                    label="CUDA Index",
                    choices=[str(i) for i in range(torch.cuda.device_count())],
                    value="0",
                    type="index",
                )

        with gr.Column():
            use_audio_prompt = gr.Checkbox(label="Use Audio Prompt")
            audio_prompt_file = gr.Audio(
                label="Audio Prompt File", type="filepath", visible=False
            )
            prompt_start_time = gr.Number(
                label="Prompt Start Time (s)",
                value=0.0,
                visible=False,
                interactive=True,
            )
            prompt_end_time = gr.Number(
                label="Prompt End Time (s)", value=30.0, visible=False, interactive=True
            )
            output_dir = gr.Textbox(label="Output Directory", value="./output")

            generate_btn = gr.Button("Generate Music")
            output_audio = gr.Audio(label="Generated Music")

    with gr.Row():
        gr.Examples(
            examples=[
                [
                    "m-a-p/YuE-s1-7B-anneal-en-cot",
                    "inspiring female uplifting pop airy vocal electronic bright vocal vocal",
                    """[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny
I know I let you down, I made mistakes
But I'm here to mend the heart I didn't break

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light
You can't fight this feeling now
I won't back down
You know you can't deny it now
I won't back down

[verse]
They might say I'm foolish, chasing after you
But they don't feel this love the way we do
My heart beats only for you, can't you see?
I won't let you slip away from me

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light
You can't fight this feeling now
I won't back down
You know you can't deny it now
I won't back down

[bridge]
No, I won't back down, won't turn around
Until you're back where you belong
I'll cross the oceans wide, stand by your side
Together we are strong

[outro]
Every road you take, I'll be one step behind
Every dream you chase, love's the tie that binds
You can't fight this feeling now
I won't back down""",
                ],
                [
                    "m-a-p/YuE-s1-7B-anneal-zh-cot",
                    "儿童 female clear vocal 动画 high-pitched vocal 声乐 有趣 打击乐器 欢乐",
                    """[verse]
疙疙瘩瘩，松松垮垮，
歪歪扭扭，坑坑洼洼。
大脑，今天再次启动咯！
行嘞~！
我是疙疙瘩瘩的大脑，
时而迷糊时而明晰。
我是疙疙瘩瘩的大脑，
哟呵！搞清楚了！
我是松松垮垮的大脑，
有时愉快有时烦闷。
我是松松垮垮的大脑，
哎~实在太烦啦！
疙疙瘩瘩，松松垮垮，
歪歪扭扭，坑坑洼洼。
哟~真不轻松呀！


[chorus]
我是晃晃的大脑，
思一思后做判定。
我是晃晃的大脑，
嘿！就选定！
我是凸凹的大脑，
壮壮身躯随我令。
我是凸凹的大脑，
哟！要警醒！
皱皱巴巴，软软绵绵，
弯弯曲曲，凹凸不平。
大脑，今天也多谢你啦！


[end]
没关系！""",
                ],
            ],
            inputs=[stage1_model, genre_txt, lyrics_txt],
            cache_examples=False,
            examples_per_page=100,
        )

    # Show/hide audio prompt options based on checkbox
    def update_prompt_times(audio_data):
        if audio_data is None:
            return gr.update(value=0.0), gr.update(value=30.0)
        return gr.update(value=0.0), gr.update(
            value=audio_data[1]
        )  # audio_data[1] contains duration

    use_audio_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[use_audio_prompt],
        outputs=[audio_prompt_file],
    )
    use_audio_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[use_audio_prompt],
        outputs=[prompt_start_time],
    )
    use_audio_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[use_audio_prompt],
        outputs=[prompt_end_time],
    )

    # Update prompt times when audio file changes
    audio_prompt_file.change(
        fn=update_prompt_times,
        inputs=[audio_prompt_file],
        outputs=[prompt_start_time, prompt_end_time],
    )

    # Connect the generate button to the generate_music function
    generate_btn.click(
        fn=generate_music,
        inputs=[
            genre_txt,
            lyrics_txt,
            stage1_model,
            stage2_model,
            max_new_tokens,
            run_n_segments,
            stage2_batch_size,
            use_audio_prompt,
            audio_prompt_file,
            prompt_start_time,
            prompt_end_time,
            output_dir,
            keep_intermediate,
            disable_offload_model,
            cuda_idx,
            rescale,
        ],
        outputs=[output_audio],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
