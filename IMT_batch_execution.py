import subprocess
import time
import pathlib
import os
import ctypes
import sys

# === Check administrator privileges ===
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    print("Administrator privileges required to run this program")
    print("Please right-click Command Prompt and select 'Run as administrator'")
    input("Press Enter to exit...")
    sys.exit()

# === Configuration section ===  "please change to your path"
imt15_exe = pathlib.Path("IMTdata/3P_COOL/3_POINT/IMT15.EXE").absolute()   # IMT15 main program path
imt16_exe = pathlib.Path("IMTdata/5P/7_POINT/IMT16.EXE").absolute()         # IMT16 main program path
ins_dir = r"IMTdata"  # .ins file directory

# === Recursively scan .ins files ===
ins_files = sorted(pathlib.Path(ins_dir).rglob("*.ins"))

if not ins_files:
    print("No .ins files found, please check the path.")
    exit()

print(f"Found {len(ins_files)} .ins files, starting execution...\n")

successful_runs = 0
total_time = 0

# === Loop through each .ins file and measure execution time ===
for ins_file in ins_files:
    print(f"Running: {ins_file}")
    
    # Select executable based on path
    if "5P" in str(ins_file):
        exe_path = str(imt16_exe)
        exe_name = "IMT16.EXE"
    else:  # 3P_COOL or 3P_heat
        exe_path = str(imt15_exe)
        exe_name = "IMT15.EXE"
    
    # Check if IMT program exists
    if not pathlib.Path(exe_path).exists():
        print(f"Program not found: {exe_path}, skipping")
        print()
        continue
    
    print(f"  Using program: {exe_name}")
    print(f"  Working directory: {ins_file.parent}")
    start_time = time.perf_counter()

    # Call imt.exe
    try:
        print(f"  Starting program: {exe_path}")
        
        # Start program and input .ins filename via stdin
        process = subprocess.Popen([exe_path], 
                                 stdin=subprocess.PIPE, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 cwd=str(ins_file.parent),
                                 text=True,
                                 shell=True)
        
        print(f"  Program started, inputting filename: {ins_file.name}")
        
        # Input .ins filename (without path), set timeout
        try:
            stdout, stderr = process.communicate(input=ins_file.name + '\n', timeout=300)
            print(f"  Program output: {stdout}")
            if stderr:
                print(f"  Error output: {stderr}")
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"Program execution timeout, skipping")
            print()
            continue
        
        if process.returncode != 0:
            print(f"  Program return code: {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, exe_path)
            
    except subprocess.CalledProcessError as e:
        print(f"Execution error, skipping: {e}")
        print()
        continue
    except FileNotFoundError as e:
        print(f"Executable file not found, skipping: {e}")
        print()
        continue
    except Exception as e:
        print(f"Unknown error, skipping: {e}")
        print()
        continue
    
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.3f} seconds")
    successful_runs += 1
    total_time += elapsed
    print()

print(f"All executions completed")
print(f"Successful runs: {successful_runs}/{len(ins_files)} files")
if successful_runs > 0:
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time: {total_time/successful_runs:.3f} seconds")
print(f"All executions completed")
print(f"Successful runs: {successful_runs}/{len(ins_files)} files")
if successful_runs > 0:
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time: {total_time/successful_runs:.3f} seconds")
