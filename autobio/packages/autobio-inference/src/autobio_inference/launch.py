import json
import os
import subprocess
import tempfile
import sys


def launch_server(fixed_args: list[str], default_extra_args: list[str]):
    extra_args = default_extra_args
    while True:
        with tempfile.NamedTemporaryFile(delete=True, mode='r') as arg_file:
            res = subprocess.run(
                [
                    *fixed_args,
                    *extra_args,
                ],
                env={
                    **os.environ,
                    "RELAUNCH_ARG_FILE": arg_file.name,
                },
            )
            if res.returncode == 101:
                # relaunch
                arg_file.seek(0)
                extra_args = json.load(arg_file)
            else:
                break


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--" in args:
        idx = args.index("--")
        fixed_args = args[:idx]
        extra_args = args[idx + 1 :]
    else:
        fixed_args = args
        extra_args = []
    launch_server(fixed_args, extra_args)
