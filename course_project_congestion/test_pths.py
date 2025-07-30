import subprocess
import re
from tabulate import tabulate
import os
from concurrent.futures import ThreadPoolExecutor
import threading

CONDA_PATH = "/home/lhc/anaconda3"
CONDA_ENV = "circuitnet"

ENABLE_MULTI_THREADING = False
ENABLE_REAL_TIME_OUTPUT = False

model_folders = [
    "congestion_double_gpdl",
    "congestion_double_gpdl_distill",
    "congestion_gpdl_1.0",
    "congestion_gpdl_0.1",
    "congestion_gpdl_0.25",
    "congestion_gpdl_0.5"
]

iter_values = [i for i in range(10000, 210000, 10000)]

results = []
if ENABLE_MULTI_THREADING:
    results_lock = threading.Lock()

def run_test(folder, iter_value):
    activate_cmd = f'source {CONDA_PATH}/bin/activate {CONDA_ENV} &&'
    command = f'{activate_cmd} python test.py --pretrained {folder}/model_iters_{iter_value}.pth'
    
    if "double" in folder.lower():
        command += ' --model_type DoubleGPDL'
    
    if ENABLE_REAL_TIME_OUTPUT:
        print(f"\n=== Testing {folder} - {iter_value} iterations ===")
    
    try:
        if ENABLE_REAL_TIME_OUTPUT:
            process = subprocess.Popen(
                command,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            for line in process.stdout:
                print(line.rstrip())
                output_lines.append(line)
            
            process.wait()
            output = ''.join(output_lines)
        else:
            output = subprocess.check_output(
                command,
                shell=True,
                executable="/bin/bash",
                stderr=subprocess.STDOUT,
                text=True
            )
        
        nrms = re.search(r"===> Avg\. NRMS: ([\d.]+)", output)
        ssim = re.search(r"===> Avg\. SSIM: ([\d.]+)", output)
        
        result = {
            "Model": folder,
            "Iterations": iter_value,
            "NRMS": float(nrms.group(1)) if nrms else "N/A",
            "SSIM": float(ssim.group(1)) if ssim else "N/A",
            "Output": output
        }
        
        if ENABLE_MULTI_THREADING:
            with results_lock:
                results.append(result)
        else:
            results.append(result)
            
    except Exception as e:
        error_msg = str(e)
        if ENABLE_REAL_TIME_OUTPUT:
            print(f"Error: {error_msg}")
        
        result = {
            "Model": folder,
            "Iterations": iter_value,
            "NRMS": "N/A",
            "SSIM": "N/A",
            "Output": error_msg
        }
        
        if ENABLE_MULTI_THREADING:
            with results_lock:
                results.append(result)
        else:
            results.append(result)

def test_model(folder):
    for iter_value in iter_values:
        run_test(folder, iter_value)

os.makedirs("test_pths_results", exist_ok=True)

if ENABLE_MULTI_THREADING:
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(test_model, model_folders)
else:
    for folder in model_folders:
        test_model(folder)

results.sort(key=lambda x: (x["Model"], int(x["Iterations"])))

table_data = []
for result in results:
    table_data.append([
        result["Model"],
        result["Iterations"],
        result["NRMS"],
        result["SSIM"]
    ])

headers = ["Model", "Iterations", "NRMS", "SSIM"]
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

with open("test_pths_results/results_summary.txt", "w") as f:
    f.write(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

for folder in model_folders:
    model_results = [r for r in results if r["Model"] == folder]
    if not model_results:
        continue
        
    with open(f"test_pths_results/{folder}_results.txt", "w") as f:
        f.write(tabulate(
            [[r["Iterations"], r["NRMS"], r["SSIM"]] for r in model_results],
            headers=["Iterations", "NRMS", "SSIM"],
            tablefmt="grid",
            floatfmt=".4f"
        ))

print("\nAll tests completed! Results saved to test_pths_results/ directory")
