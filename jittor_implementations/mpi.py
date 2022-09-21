import os
import jittor as jt

def fork_with_mpi(num_procs=4):
    import sys
    if jt.in_mpi:
        # you can mult other process output
        if jt.rank != 0:
            sys.stdout = open("/dev/null", "w")
        return
    else:
        print(sys.argv)
        cmd = " ".join(["mpirun", "-np", str(num_procs), sys.executable] + sys.argv)
        print("[RUN CMD]:", cmd)
        os.system(cmd)
        # exit(0)