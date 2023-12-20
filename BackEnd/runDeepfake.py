import subprocess

# Path to the Python interpreter in your virtual environment
venv_python = "C:\\Users\\Asus\\anaconda3\\envs\\deepfakeenv\\python.exe"

# Command you want to run inside the virtual environment
script_path = "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\Deepfake\\Audio auto Encoder to Video\\script.py"
face = "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\uploads\\inputVideo.mp4"
speech = "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\out\\output.wav"
command = [venv_python, script_path, "--face=" + face, "--speech=" + speech]

# Execute the command
subprocess.run(command)
