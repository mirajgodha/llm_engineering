import os
import cv2
import torch
import random
import pickle
import datetime
import platform
import re
import gc
import time
import numpy as np
from PIL import Image
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.quantizers import PipelineQuantizationConfig
from moviepy import VideoFileClip, AudioFileClip, afx
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from diffusers.utils import export_to_video

VIDEO_CATEGORY_ID = "22"  # People & Blogs
VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_NUM_FRAMES =  512,512, 129 # ~4 sec at 15 fps
HF_TOKEN=""
VIDEO_NUM_PROMPTS = 4
#Models
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
GPT2="openai-community/gpt2"


# -------------------
# Hugging Face Login
# -------------------
def login_huggingface():
    login(HF_TOKEN)

# -------------------
# Generate Series of Prompts with Same Character
# -------------------
def generate_prompt_with_character(character):
    actions = [
        "dances on a trampoline",
        "juggles glowing balls",
        "paints the sky with colors",
        "builds a magical fort",
        "flies a candy spaceship",
        "runs with sparkle shoes",
        "sings in a parade of cupcakes",
        "plays a tune on star-shaped drums",
        "throws glitter confetti everywhere"
    ]
    settings = [
        "on the moon made of cheese",
        "under the jelly ocean",
        "in a forest of lollipops",
        "inside a toy rocket",
        "at a rainbow zoo",
        "on a trampoline planet",
        "in a land of bouncing castles",
        "in space where planets sing",
        "inside a cloud-shaped classroom"
    ]
    styles = [
        "in cartoon style",
        "as a 3D animation",
        "in pixel art",
        "in stop-motion",
        "in anime style",
        "as a claymation",
        "with glowing neon colors",
        "in watercolor splashes",
        "as a doodle drawing"
    ]
    return f"{character} {random.choice(actions)} {random.choice(settings)} {random.choice(styles)}"



# -------------------
# Model Setup
# -------------------
def load_model(model_id):
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

def get_pipeline():
    # quantize weights to int4 with bitsandbytes
    # pipeline_quant_config = PipelineQuantizationConfig(
    #     quant_backend="bitsandbytes_4bit",
    #     quant_kwargs={
    #         "load_in_4bit": True,
    #         "bnb_4bit_quant_type": "nf4",
    #         "bnb_4bit_compute_dtype": torch.bfloat16
    #     },
    #     components_to_quantize=["transformer"]
    # )
    #
    # pipeline = HunyuanVideoPipeline.from_pretrained(
    #     "hunyuanvideo-community/HunyuanVideo",
    #     quantization_config=pipeline_quant_config,
    #     torch_dtype=torch.bfloat16,
    # )
    #
    # # model-offloading and tiling
    # pipeline.enable_model_cpu_offload()
    # pipeline.vae.enable_tiling()
    #
    # # torch.compile
    # pipeline.transformer.to(memory_format=torch.channels_last)
    # pipeline.transformer = torch.compile(
    #     pipeline.transformer, mode="max-autotune", fullgraph=True
    # )
    #

    # video_model_id = "hunyuanvideo-community/HunyuanVideo"
    # quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    #
    # transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    #     video_model_id,
    #     subfolder="transformer",
    #     quantization_config=quant_config,
    #     torch_dtype=torch.bfloat16,
    # )
    #
    # pipe = HunyuanVideoPipeline.from_pretrained(
    #     video_model_id,
    #     transformer=transformer,
    #     torch_dtype=torch.float16
    # )
    # pipe.vae.enable_tiling()
    # pipe.enable_model_cpu_offload()

    # quantize weights to int4 with bitsandbytes
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16
        },
        components_to_quantize=["transformer"]
    )

    pipeline = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        quantization_config=pipeline_quant_config,
        torch_dtype=torch.bfloat16,
    )

    # model-offloading and tiling
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_tiling()

    return pipeline


# -------------------
# Creative Prompts Generation
# -------------------
# -------------------
# Creative Prompts Generation (Hardcoded)
# -------------------
def generate_creative_elements():
    creative_elements = {
        "characters": [
            "A brave lion knight",
            "A wise owl librarian",
            "A clumsy panda astronaut",
            "A cheerful monkey pirate",
            "A shy turtle magician",
            "A giggling giraffe ballerina",
            "A sleepy koala chef",
            "A speedy rabbit pilot",
            "A playful puppy detective",
            "A curious cat scientist",
            "A surfing crocodile",
            "A penguin pop star",
            "A hamster firefighter",
            "A dancing duck DJ",
            "A singing elephant",
            "A racing raccoon",
            "A painter parrot",
            "A robot squirrel",
            "A ninja frog",
            "A basketball-playing bear",
            "A unicorn zebra",
            "A banana-suit snake",
            "A jellyfish artist",
            "A walrus wizard",
            "A llama rockstar"
        ],
        "actions": [
            "Dances on a trampoline",
            "Juggles glowing balls",
            "Paints the sky with colors",
            "Sings with a bird choir",
            "Slides down a rainbow",
            "Builds castles of marshmallows",
            "Swims in a chocolate river",
            "Plays hide and seek in clouds",
            "Blows giant soap bubbles",
            "Flies with a balloon backpack",
            "Rides a rollercoaster made of vines",
            "Eats moon-shaped cookies",
            "Bounces on jelly hills",
            "Spins in cotton candy tornadoes",
            "Drives a toy train in the sky",
            "Climbs a beanstalk to the stars",
            "Draws with magic crayons",
            "Zooms through space in a fruit rocket",
            "Sleds on whipped cream mountains",
            "Jumps into a book of dreams",
            "Plays hopscotch on planets",
            "Sings underwater with mermaids",
            "Skates on a frozen juice lake",
            "Does yoga with flamingos",
            "Picks candy fruits from rainbow trees"
        ],
        "settings": [
            "On the moon made of cheese",
            "Under the jelly ocean",
            "In a forest of lollipops",
            "Inside a cloud castle",
            "Atop a floating island",
            "In a cupcake mountain range",
            "In a garden of talking flowers",
            "Inside a rainbow tunnel",
            "On a flying pirate ship",
            "In a playground of stars",
            "In a bubble village",
            "On a marshmallow planet",
            "Inside a soda volcano",
            "In a cotton candy desert",
            "At a glowing coral reef",
            "In a toy factory",
            "Inside a storybook world",
            "On a bouncing cloud field",
            "In a giant teacup city",
            "Inside a jungle of musical vines",
            "In an upside-down treehouse",
            "On a caramel slide",
            "Inside a glitter dome",
            "On a banana boat island",
            "In a sparkly snowflake town"
        ],
        "styles": [
            "Cartoon",
            "3D",
            "Watercolor",
            "Paper cutout",
            "Stop motion",
            "Chalkboard",
            "Pixel art",
            "Crayon sketch",
            "Clay animation",
            "Neon lights",
            "Glow in the dark",
            "Origami",
            "Mixed media",
            "Retro comic",
            "Stained glass",
            "Minimalist",
            "Graffiti",
            "Storybook",
            "Felt puppet",
            "Soft pastel",
            "Hand-drawn",
            "Sticker art",
            "Pop-up book",
            "Plasticine",
            "Fantasy brush"
        ]
    }
    return creative_elements

def generate_prompts(creative_elements, n=1):
    def gen_prompt(char, actions, settings, styles):
        return f"{char}, {random.choice(actions)}. Background: {random.choice(settings)}. Video Style: {random.choice(styles)}"

    main_character = random.choice(creative_elements["characters"])
    prompts = [gen_prompt(main_character, creative_elements["actions"], creative_elements["settings"], creative_elements["styles"]) for _ in range(n)]
    return main_character, prompts

# -------------------
# Video Generation
# -------------------
def generate_video_segments(pipe, prompts, raw_video_file):

    all_frames = []
    for i, prompt in enumerate(prompts):
        print(f"🎬 Generating segment {i+1}/{len(prompts)}: {prompt}")
        output = pipe(
            prompt=prompt,
            height=VIDEO_HEIGHT,
            width=VIDEO_WIDTH,
            num_frames=VIDEO_NUM_FRAMES,  # ~4 sec at 15 fps
            num_inference_steps=30
        ).frames[0]
        all_frames.extend(output)
        export_to_video(all_frames, raw_video_file, fps=15)
    return all_frames

def clean_youtube_tags(tag_string, char_limit=500):
    # Step 1: Split and remove '#'
    raw_tags = tag_string.replace('#', '').split(',')

    # Step 2: Remove duplicates, preserve order
    seen = set()
    unique_tags = []
    for tag in raw_tags:
        tag = tag.strip()
        if tag and tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    # Step 3: Limit to 500 characters total
    final_tags = []
    total_chars = 0
    for tag in unique_tags:
        if total_chars + len(tag) + 2 > char_limit:  # +2 for quotes and comma spacing
            break
        final_tags.append(tag)
        total_chars += len(tag) + 2

    return final_tags

def get_title(joined_prompts, tokenizer, model):
    messages = [
        {"role": "system",
         "content": "I have created a kids small youtube video. You will help me to generate metadata for my new youtube video"},
        {"role": "user",
         "content": f"Suggest a catchy title for a youtube video for the given prompts. "
                    f"\nPrompts:\n{joined_prompts} "
                    f"\n E.g. Output: "
                    f"\n1. **Monkey Business: Banana Boat Adventures** "
                    f"\n2. **Sweet Swashbuckler: A Monkey Pirate's Cotton Candy Quest** "
                    f"\n3. **Rollercoaster Monkey: A Jungle of Wonder** "
                    f"\n4. **Banana Bandit: Spinning in Sugar and Spice**"}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, padding=True).to("cuda")
    outputs = model.generate(inputs, max_new_tokens=80)

    decoded_output = tokenizer.decode(outputs[0])

    match = re.search(r'1\.?\s+\*\*(.*?)\*\*', decoded_output)
    if match:
        first_suggestion = match.group(1)
    else:
        first_suggestion = "Kids Animation Video"
        print(f"✅ Title Unprocessed: {decoded_output}")

    return first_suggestion

def get_description(joined_prompts, tokenizer, model):
    messages = [
        {"role": "system",
         "content": "I have created a kids small youtube video. You will help me to generate metadata for my new youtube video"},
        {"role": "user",
         "content": f"Suggest a detailed description for a youtube video, for the given prompts: \nPrompts:\n{joined_prompts} "}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, padding=True).to("cuda")
    outputs = model.generate(inputs, max_new_tokens=150)

    decoded_output = tokenizer.decode(outputs[0])

    # Extract description after "**Description:**"
    description_match = re.search(r"\*Description:\*\s*(.+)", decoded_output, re.DOTALL)

    if description_match:
        raw_desc = description_match.group(1)
        # Remove unwanted characters at start (like quotes, dashes, newlines)
        raw_desc = re.sub(r"^[^a-zA-Z0-9]+", "", raw_desc)
        # Remove extra blank lines and strip whitespaces
        lines = [line.strip() for line in raw_desc.splitlines() if line.strip()]
        cleaned_desc = " ".join(lines)

    if not description_match:
        cleaned_desc = "Kids Animation Video"
        print(f"✅ Description Unprocessed: {decoded_output}")

    return cleaned_desc

def get_tags(joined_prompts, tokenizer, model):
    messages = [
        {"role": "system",
         "content": "I have created a kids small youtube video. You will help me to generate metadata for my new youtube video"},
        {"role": "user",
         "content": f"Suggest at least 5-10 hashtags for a youtube video which increases seo, for the given prompts: \nPrompts:\n{joined_prompts} "}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, padding=True).to("cuda")
    outputs = model.generate(inputs, max_new_tokens=300)

    decoded_output = tokenizer.decode(outputs[0])

    # Extract hashtags using regex
    hashtags = re.findall(r"#\w+", decoded_output)

    # Convert to CSV string
    csv_hashtags = ",".join(hashtags)

    if not csv_hashtags or csv_hashtags=="" or csv_hashtags==",":
        print(f"✅ Tags Unprocessed: {decoded_output}")
        csv_hashtags = "animation,kids,funny"

    return clean_youtube_tags(csv_hashtags)
# -------------------
# Metadata Generation
# -------------------
def generate_metadata(prompts, tokenizer, model):
    joined_prompts = "\n".join(prompts)

    title = get_title(joined_prompts, tokenizer, model)
    description = get_description(joined_prompts, tokenizer, model)
    tags = get_tags(joined_prompts, tokenizer, model)

    print(f"✅ Title: {title}")
    print(f"✅ Description: {description}")
    print(f"✅ Tags: {tags}")

    return title, description, tags

# -------------------
# Save Video
# -------------------
def save_video(frames_nested, file_name):
    frames = [frame for segment in frames_nested for sequence in segment for frame in sequence if isinstance(frame, Image.Image)]
    if not frames:
        raise ValueError("No frames found to write into video. Please check generation pipeline.")

    width, height = frames[0].size
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(file_name, fourcc, fps, (width, height))

    for frame in frames:
        np_frame = np.array(frame)
        bgr_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()
    print(f"✅ Video saved as {file_name}")

# -------------------
# Add Audio to Video
# -------------------
def add_audio_to_video(video_file, audio_file, output_file):
    video = VideoFileClip(video_file)
    audio = AudioFileClip(audio_file)
    video_duration = video.duration

    if audio.duration < video_duration:
        audio = audio.fx(afx.audio_loop, duration=video_duration)
    elif audio.duration > video_duration:
        audio = audio.with_duration(video_duration)

    final_video = video.with_audio(audio)
    final_video.write_videofile(output_file, codec="libx264", audio_codec="aac")
    print(f"🎵 Audio added and saved to {output_file}")

# -------------------
# YouTube Upload
# -------------------
def authenticate_youtube():
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    CREDENTIALS_FILE = "client_secrets.json"
    TOKEN_FILE = "token.pkl"
    credentials = None

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as f:
            credentials = pickle.load(f)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            system = platform.system().lower()
            if system == "darwin":  # macOS
                credentials = flow.run_local_server(port=0)
            else:  # Linux, Windows, etc.
                credentials = flow.run_console()
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(credentials, f)

    return build("youtube", "v3", credentials=credentials)


def upload_video(youtube, video_path, title, description, tags):
    media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": VIDEO_CATEGORY_ID,
                "tags": tags
            },
            "status": {
                "privacyStatus": "public",
                "madeForKids": True
            },
        },
        media_body=media
    )

    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"Uploading: {int(status.progress() * 100)}%")
        except Exception as e:
            print(f"❌ Error during upload: {e}")
            break

    if response:
        print(f"✅ Uploaded: https://youtu.be/{response['id']}")

# -------------------
# Main Execution
# -------------------
def main(tokenizer, model,pipe):


    print("Generating creative elements..............................")
    creative_elements = generate_creative_elements()
    print("Generated creative elements..............................")

    print("Generating prompts.....................................")
    main_char, prompts = generate_prompts(creative_elements, VIDEO_NUM_PROMPTS)

    title, desc, tags = generate_metadata(prompts, tokenizer, model)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = title.lower().replace(" ", "_").replace("/", "-")[:50]
    raw_video_file = f"{slug}_{timestamp}.mp4"
    final_video_file = f"{slug}_{timestamp}_with_audio.mp4"
    #remove any special characters from final_video_file
    final_video_file = re.sub(r'[<>:"/\\|?*]', '', final_video_file)

    #Clean memory
    # del model, tokenizer
    # gc.collect()
    # torch.cuda.empty_cache()


    for i, prompt in enumerate(prompts):
        print(f"\n🎬 Prompt {i+1}: {prompt}")

    print("Generating video segments..............................")
    all_frames = generate_video_segments(pipe, prompts, raw_video_file)

    # print(f"Saving video to {raw_video_file}......................")
    # save_video(all_frames, raw_video_file)
    print(f"Adding audio to video and saving to {final_video_file}..............")
    # random number 1 to 10
    i = random.randint(1, 10)
    audio_file = f"audio_{i}.mp3"
    add_audio_to_video(raw_video_file, audio_file, final_video_file)

    print("Authenticating to YouTube.............................")
    youtube = authenticate_youtube()
    print("Uploading video to YouTube..........................")
    upload_video(youtube, final_video_file, title, desc, tags)


if __name__ == "__main__":
    login_huggingface()
    #print loading time of models
    start_time = time.time()
    model_id = LLAMA
    print(f"Loading {model_id}...")
    tokenizer, model = load_model(model_id)
    pipe = get_pipeline()
    end_time = time.time()
    print(f"Total models load time is: {end_time - start_time:.2f} seconds")
    # loop 10 times
    for i in range(10):
        try:
            # Calculate totalt time taken by function
            start_time = time.time()
            main(tokenizer, model,pipe)
            end_time = time.time()
            print(f"Total time taken: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Error: {e}")
