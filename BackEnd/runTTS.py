import subprocess

# Path to the Python interpreter in your virtual environment
venv_python = "C:\\Users\\Asus\\anaconda3\\envs\\ttsenvv\\python.exe"

# Command you want to run inside the virtual environment
script_path = "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\TTS\\Capstone 2\\script_final_latest.py"
text_file = "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\outputs\\output.txt"
video_file = "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\uploads\\inputVideo.mp4"
command = [venv_python, script_path, "--text_file=" + text_file, "--video_file=" + video_file]

# Execute the command
subprocess.run(command)
