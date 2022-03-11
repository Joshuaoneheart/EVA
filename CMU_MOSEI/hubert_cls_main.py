import os
out_file = "hubert_output.txt"
command = f"touch {out_file}"
os.system(command)
for i in range(4):
    print(f"Running round {i}")
    command = f"time python3 hubert_cls.py | tee -a {out_file}"
    os.system(command)
