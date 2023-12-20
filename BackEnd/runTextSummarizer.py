import subprocess

# Path to the Python interpreter in your virtual environment
venv_python = 'C:\\Users\\Asus\\Desktop\\Capstone\\TextSummarizer\\myenv\\Scripts\\python.exe'

# Command you want to run inside the virtual environment
command = 'run.py'

# Additional command-line arguments (if needed)
# arguments = ['arg1', 'arg2']

# Construct the command
cmd = [venv_python, command]

# Execute the command
subprocess.run(cmd)
