from clearml import Task

if __name__ == "__main__":
    task: Task = Task.init(project_name="demo", task_name="slurm_demo")

    # Slurm configuration
    slurm_params = {
        "memory": "16GB",
        "time": "01:00:00",
        "cpu": 4,
        "gpu": 0,
        "log_dir": "/vast/wlp9800/logs",
        "singularity_overlay": "",
        "singularity_binds": "",
        "container_source": {"sif_path": "/scratch/wlp9800/images/devenv-cpu.sif", "type": "sif_path"},
        "use_singularity": True,
        "setup_commands": "module load python/intel/3.8.6",
        "skip_python_env_install": True,
    }
    # slurm_params = {
    #     "memory": "16GB",
    #     "time": "01:00:00",
    #     "cpu": 4,
    #     "gpu": 0,
    #     "log_dir": "/vast/wlp9800/logs",
    #     "singularity_overlay": "",
    #     "singularity_binds": "",
    #     "container_source": {},
    #     "use_singularity": False,
    #     "setup_commands": "module load python/intel/3.8.6",
    #     "skip_python_env_install": False,
    # }
    task.connect(slurm_params, name="slurm")

    task.execute_remotely(queue_name="slurm_demo", exit_process=True)

    print("Hello World")
