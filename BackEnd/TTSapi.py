import requests
import argparse
from pydub import AudioSegment
import io

CHUNK_SIZE = 1024

# Define the URLs for both male and female voices
FEMALE_URL = "https://api.elevenlabs.io/v1/text-to-speech/7aRcKuo7XjNH7WNhsxv9"
MALE_URL = "https://api.elevenlabs.io/v1/text-to-speech/ItuM0Is7CQaPOkO8QpHY"

headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": "40703485f3e9151510ccd64558a15ae4"
}

def main():
    parser = argparse.ArgumentParser(description="Convert text to speech using an API")
    parser.add_argument("--text_file", type=str, help="Path to the text file")
    parser.add_argument("--gender", choices=["MALE", "FEMALE"], help="Gender of the voice")

    args = parser.parse_args()

    if not args.text_file:
        print("Please provide a path to the text file using --text_file.")
        return

    # Determine the URL based on the gender argument
    if args.gender == "FEMALE":
        url = FEMALE_URL
    else:
        url = MALE_URL

    try:
        with open(args.text_file, 'r') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"File not found: {args.text_file}")
        return

    print(text)
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        audio_data = response.content
        audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio.export("out\\output.wav", format="wav")
        print("Audio file 'output.wav' generated successfully.")
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")

if __name__ == "__main__":
    main()
