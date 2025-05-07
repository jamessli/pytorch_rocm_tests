import os
import subprocess
import re
import torch


class PyTorchTests:
    def __init__(self, args, config):
        self.name         = 'pytorchtests'
        self.binary_name  = 'PyTorchTests'
        self.force_fp_16  = config['fp_16_enabled']
        self.models       = config['models']
        
        self.gpus        = self._get_gpus()
        self.world_size  = len(self.gpus)
        
        self.threshold   = getattr(args, 'threshold', 1.0)
        
        self._golden_data = { "all_tests": 0 }
        
        self.log_path = getattr(args, 'log_path', os.path.join(os.getcwd(), 'logs'))
        os.makedirs(self.log_path, exist_ok=True)
        
        self.env = os.environ.copy()
        here = os.path.dirname(os.path.realpath(__file__))
        self.env["PATH"] = os.path.join(here, '..', 'venv', 'bin')
        print("Using PATH:", self.env["PATH"])

    def _get_gpus(self):
        """Return a list of available GPU IDs (0..N-1)."""
        try:
            count = torch.cuda.device_count()
        except Exception:
            count = 0
        return list(range(count)) if count > 0 else [0]

    def write_to_log(self, result, header, append=False):
        """Write stdout/stderr of a benchmark run into the log file."""
        mode = 'a' if append else 'w'
        logfile = os.path.join(self.log_path, f"{self.name}.log")
        with open(logfile, mode) as f:
            f.write(header)
            f.write(result.stdout or "")
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

    def run(self):
        """Launch each model test distributed across all GPUs."""
        is_first = True
        cmd_tpl = (
            "python3 pytorch-micro-benchmarking/"
            "micro_benchmarking_pytorch.py "
            "--device_ids={gpu} "
            "--network {model} "
            "--distributed_dataparallel "
            "--rank {gpu} "
            "--world-size {world} "
            "--dist-backend nccl "
            "--dist-url tcp://127.0.0.1:4332"
        )

        for model in self.models:
            # build one backgrounded command per GPU
            parts = [
                cmd_tpl.format(gpu=g, model=model, world=self.world_size)
                for g in self.gpus
            ]
            full_cmd = " & ".join(parts)
            
            try:
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=self.env
                )
                self.write_to_log(
                    result,
                    header=f"Model Tested = {model}\n",
                    append=not is_first
                )
            except Exception as e:
                print(f"PyTorch tests failed with error: {e}")
            
            is_first = False

    def comparator(self):
        """Parse the latest log and compare throughput against golden data."""
        latest = max(os.listdir(self.log_path))
        path   = os.path.join(self.log_path, latest)
        pattern = re.compile(
            r"--------Overall \(all ranks\).*?--------\n"
            r"Num devices:\s*(\d+)\n"
            r"Dtype:\s*(\w+)\n"
            r"Mini batch size \[img\]\s*:\s*(\d+)\n"
            r"Time per mini-batch\s*:\s*([\d\.]+)\n"
            r"Throughput \[img/sec\]\s*:\s*([\d\.]+)",
            re.S
        )

        with open(path, "r") as f:
            data = f.read()

        try:
            # take the last match block
            num, dtype, batch, tpb, thrpt = pattern.findall(data)[-1]
            thrpt_val = round(float(thrpt), 2)
            
            # find which model was tested
            m = re.search(r"Model Tested\s*=\s*(.+)", data)
            model_name = m.group(1).strip() if m else "<unknown>"

            baseline = self._golden_data['all_tests']
            if thrpt_val < baseline * self.threshold:
                print(f"{model_name} throughput FAILED: {thrpt_val} img/sec")
            else:
                print(f"{model_name} throughput PASSED: {thrpt_val} img/sec")

        except (IndexError, ValueError) as e:
            print(f"Log parsing failed ({e}); file may be incomplete.")