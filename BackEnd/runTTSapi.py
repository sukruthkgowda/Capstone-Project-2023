import subprocess

# Command you want to run inside the virtual environment
script_path = "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\TTSapi.py"
text_file = "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\outputs\\output.txt"
gend = "MALE"
command = ["python", script_path, "--text_file=" + text_file, "--gender=" + gend]

# Execute the command
subprocess.run(command)
