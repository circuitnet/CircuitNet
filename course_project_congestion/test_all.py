#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import threading
import queue

def is_tqdm_line(line):
    """
    Check if a line is from tqdm progress bar
    """
    tqdm_indicators = [
        '%|',           # Progress percentage
        'it/s',         # Iterations per second
        '█',            # Progress bar character
        'ETA:',         # Estimated time
        '/s]',          # Speed indicator
        '[<',           # Progress bar start
        '\r'            # Carriage return
    ]
    
    # Check if line contains tqdm indicators
    for indicator in tqdm_indicators:
        if indicator in line:
            return True
    
    # Check for progress pattern like "1%|" or "100%|"
    import re
    if re.match(r'\s*\d+%\|', line):
        return True
        
    # Check if line is mostly spaces and special characters (typical for progress bars)
    if len(line) > 10 and len(line.replace(' ', '').replace('|', '').replace('█', '').replace('▏', '').replace('▎', '').replace('▍', '').replace('▌', '').replace('▋', '').replace('▊', '').replace('▉', '')) < len(line) * 0.3:
        return True
    
    return False

def run_command_with_live_output(cmd, output_file=None, timeout=600):
    """
    Run command with live output display and optional file capture
    """
    start_time = datetime.now()
    
    try:
        # Set environment to disable tqdm progress bars
        env = os.environ.copy()
        env['TQDM_DISABLE'] = '1'
        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=os.getcwd(),
            env=env  # Pass the modified environment
        )
        
        output_lines = []
        
        # Read output line by line and display in real time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                
                # Filter out tqdm progress bar lines
                if is_tqdm_line(line):
                    # Still store for file but don't display
                    if output_file:
                        output_lines.append(line)
                    continue
                
                # Display to console immediately
                print(line)
                # Store for file writing if needed
                if output_file:
                    output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.poll()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Write to file if specified
        if output_file:
            output_content = []
            output_content.append(f"# Command Output")
            output_content.append(f"**Command:** {' '.join(cmd)}")
            output_content.append(f"**Executed at:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            output_content.append(f"**Duration:** {duration:.2f} seconds")
            output_content.append(f"**Return code:** {return_code}")
            output_content.append("")
            output_content.append("## Output:")
            output_content.append("```")
            output_content.extend(output_lines)
            output_content.append("```")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_content))
        
        return return_code, duration, '\n'.join(output_lines)
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Error running command: {e}")
        return -1, duration, str(e)

def run_test_batch():
    """
    Run all test commands and save outputs to separate files
    """
    
    # Create output directory
    output_dir = Path("all_test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Define test commands with their corresponding output file names
    test_commands = [
        {
            "cmd": ["python", "test.py", "--pretrained", "./congestion.pth"],
            "output_file": "test_congestion_pretrained.txt",
            "description": "Pretrained congestion model test",
            "capture_output": True
        },
        {
            "cmd": ["python", "test.py", "--pretrained", "./congestion_gpdl_1.0/model_iters_200000.pth"],
            "output_file": "test_gpdl_1.0.txt",
            "description": "GPDL model with 100% data",
            "capture_output": True
        },
        {
            "cmd": ["python", "test.py", "--pretrained", "./congestion_gpdl_0.5/model_iters_200000.pth"],
            "output_file": "test_gpdl_0.5.txt",
            "description": "GPDL model with 50% data",
            "capture_output": True
        },
        {
            "cmd": ["python", "test.py", "--pretrained", "./congestion_gpdl_0.25/model_iters_200000.pth"],
            "output_file": "test_gpdl_0.25.txt",
            "description": "GPDL model with 25% data",
            "capture_output": True
        },
        {
            "cmd": ["python", "test.py", "--pretrained", "./congestion_gpdl_0.1/model_iters_200000.pth"],
            "output_file": "test_gpdl_0.1.txt",
            "description": "GPDL model with 10% data",
            "capture_output": True
        },
        {
            "cmd": ["python", "test_quant.py", "--pretrained", "./congestion_gpdl_1.0/model_iters_200000.pth", "--log_dir", str(output_dir)],
            "output_file": None,
            "description": "INT8 quantization test",
            "capture_output": False
        },
        {
            "cmd": ["python", "test_quant.py", "--pretrained", "./congestion_gpdl_1.0/model_iters_200000.pth", "--quant_bits", "4", "--log_dir", str(output_dir)],
            "output_file": None,
            "description": "INT4 quantization test",
            "capture_output": False
        },
        {
            "cmd": ["python", "test.py", "--pretrained", "./congestion_double_gpdl/model_iters_200000.pth", "--model_type", "DoubleGPDL"],
            "output_file": "test_double_gpdl.txt",
            "description": "DoubleGPDL enhanced model test",
            "capture_output": True
        },
        {
            "cmd": ["python", "test.py", "--pretrained", "./congestion_double_gpdl_distill/model_iters_200000.pth", "--model_type", "DoubleGPDLdistill"],
            "output_file": "test_double_gpdl_distill.txt",
            "description": "DoubleGPDLdistill enhanced model test",
            "capture_output": True
        }
    ]
    
    print("="*60)
    print("Starting batch testing...")
    print(f"Results will be saved to: {output_dir.absolute()}")
    print("Note: Progress bars are filtered for cleaner output")
    print("="*60)
    
    total_tests = len(test_commands)
    
    for i, test_config in enumerate(test_commands, 1):
        cmd = test_config["cmd"]
        output_file = test_config["output_file"]
        description = test_config["description"]
        capture_output = test_config["capture_output"]
        
        print(f"\n{'='*60}")
        print(f"[{i}/{total_tests}] Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        
        if capture_output and output_file:
            output_path = output_dir / output_file
            print(f"Output file: {output_file}")
        else:
            print("Output: Direct to log files (not captured)")
        
        print("-" * 60)
        
        try:
            if capture_output and output_file:
                # Run with live output and file capture
                return_code, duration, output_text = run_command_with_live_output(
                    cmd, output_path, timeout=600
                )
                
                # Extract metrics from output
                nrms = ssim = emd = "N/A"
                lines = output_text.split('\n')
                for line in lines:
                    if "Avg. NRMS:" in line:
                        nrms = line.split("Avg. NRMS:")[-1].strip()
                    elif "Avg. SSIM:" in line:
                        ssim = line.split("Avg. SSIM:")[-1].strip()
                    elif "Avg. EMD:" in line:
                        emd = line.split("Avg. EMD:")[-1].strip()
                
                print("-" * 60)
                if return_code == 0:
                    print(f"✅ SUCCESS - Duration: {duration:.2f}s")
                    if nrms != "N/A":
                        print(f"   NRMS: {nrms}, SSIM: {ssim}, EMD: {emd}")
                else:
                    print(f"❌ FAILED - Return code: {return_code}")
            
            else:
                # Run quantization tests with live output but no file capture
                return_code, duration, _ = run_command_with_live_output(
                    cmd, None, timeout=600
                )
                
                print("-" * 60)
                if return_code == 0:
                    print(f"✅ SUCCESS - Duration: {duration:.2f}s")
                    print("   Detailed logs saved to log files")
                else:
                    print(f"❌ FAILED - Return code: {return_code}")
            
        except KeyboardInterrupt:
            print(f"\n❌ INTERRUPTED - Test was stopped by user")
            break
            
        except Exception as e:
            print(f"❌ ERROR - {str(e)}")
            
            if capture_output and output_file:
                error_content = [
                    f"# {description}",
                    f"**Command:** {' '.join(cmd)}",
                    f"**Status:** ERROR",
                    f"**Executed at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "## Error:",
                    str(e)
                ]
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(error_content))
    
    print("\n" + "="*60)
    print("Batch testing completed!")
    print(f"Results saved to: {output_dir.absolute()}")
    print("For quantization test details, check the log files in the output directory.")
    print("="*60)

if __name__ == "__main__":
    try:
        run_test_batch()
    except KeyboardInterrupt:
        print("\n❌ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)