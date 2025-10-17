from clearml import Task

if __name__ == "__main__":
    task: Task = Task.init(project_name="demo", task_name="slurm_demo")

    # Slurm configuration
    slurm_params = {
        "memory": "16GB",
        "time": "06:00:00",
        "cpu": 4,
        "gpu": 0,
        "log_dir": "/vast/wlp9800/logs",
        "setup_commands": "module load python/intel/3.8.6",
        "singularity_overlay": "",
        "singularity_binds": "",
        "container_source": {},
        "use_singularity": False,
        "skip_python_env_install": False,
    }
    task.connect(slurm_params, name="slurm")

    task.execute_remotely(queue_name="slurm_deno", exit_process=True)

    print("Hello World")
