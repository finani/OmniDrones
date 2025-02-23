Workstation Installation
========================

From Isaac Sim 4.0 release, it is possible to install Isaac Sim using pip.
Although Isaac Sim recommend creating a virtual environment, we recommend using a separate conda environment which is more flexible.

.. note::

    `Miniconda <https://docs.anaconda.com/miniconda/>`_ .

.. seealso::

    `Managing Conda Environments <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux>`_ .

Create a new conda environment for IsaacSim. Replace "env_isaaclab" with your desired name

.. code-block:: bash

    conda create -n env_isaaclab python=3.10 -y
    conda activate env_isaaclab

Upgrade pip

.. code-block:: bash

    pip install --upgrade pip


Install torch based on the CUDA version available on your system.

.. code-block:: bash

    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

Install Isaac Sim

.. code-block:: bash

    pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

To verify the installation, run

.. code-block:: bash

    python -c "from isaacsim.simulation_app import SimulationApp"
    # Which torch is being used
    python -c "import torch; print(torch.__path__)"

To use the internal libraries included with the extension please set the following environment variables to your ``~/.bashrc`` or ``~/.zshrc``:

.. code-block:: bash

    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.10/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib

.. code-block:: bash

    # Run Isaac Sim
    isaacsim isaacsim.exp.full.kit

The next step is to install `Isaac Lab <https://github.com/isaac-sim/IsaacLab>`_ .

Install dependencies.

.. code-block:: bash

    sudo apt install cmake build-essential

Clone Isaac Lab and install it.

.. code-block:: bash

    git clone git@github.com:isaac-sim/IsaacLab.git
    cd IsaacLab
    ./isaaclab.sh --install

To verify the installation, run

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

Finally, install **OmniDrones** in editable mode (which automatically installs other required dependencies):

.. code-block:: bash

    # at OmniDrones/
    pip install -e .

To verify the installation, run

.. code-block:: bash

    cd scripts
    python train.py algo=ppo headless=true wandb.entity=YOUR_WANDB_ENTITY

In general, YOUR_WANDB_ENTITY is your wandb ID.
If you don't want to add arguments every time, edit ``scripts/train.yaml``







If you encounter the following error,
try `TypeError: ArticulationView.get_world_poses() got an unexpected keyword argument 'usd' <troubleshooting.html#typeerror-articulationview-get-world-poses-got-an-unexpected-keyword-argument-usd>`_ .

.. code-block:: bash

    File "/${HOME}/.local/share/ov/pkg/isaac-sim-4.1.0/exts/omni.isaac.core/omni/isaac/core/prims/xform_prim_view.py", line 189, in __init__
        default_positions, default_orientations = self.get_world_poses(usd=usd)
    TypeError: ArticulationView.get_world_poses() got an unexpected keyword argument 'usd'

Developer Guide: Working with VSCode
------------------------------------

To enable features like linting and auto-completion with VSCode Python Extension, we need to let the extension recognize the extra paths we added during the setup process.

Create a file ``.vscode/settings.json`` at your workspace if it is not already there.

After activating the conda environment, run

.. code:: console

    printenv > .vscode/.python.env

and edit ``.vscode/settings.json`` as:

.. code:: json

    {
        // ...
        "python.envFile": "${workspaceFolder}/.vscode/.python.env",
    }

Developer Guide: Python Environments
------------------------------------

.. list-table:: Python Environments
    :widths: 25 25 25 25 25 25
    :header-rows: 1

    * -
      - Isaac Sim 2022.*
      - Isaac Sim 2023.*
      - Isaac Sim 4.0, 4.1, 4.2
      - Isaac Lab 1.*
      - Isaac Sim 4.5, Isaac Lab 2.0
    * - python
      - 3.7
      - 3.10
      - 3.10
      - 3.10
      - 3.10
    * - pytorch
      - 1.10.0+cu113
      - 2.0.1+cu118
      - 2.2.2+cu118
      - 2.2.2+cu118
      - 2.5.1+cu118 or 2.5.1+cu121
    * - torchrl
      -
      - 0.1.1
      - 0.3.1
      - 0.3.1
      - 0.3.1
    * - tensordict
      -
      - 0.1.1
      - 0.3.2
      - 0.3.2
      - 0.3.2
      - 0.6.2

Developer Guide: Test Run
-------------------------

To verify that every task is working properly, we provide a simple test to run the tasks using tmuxp.

Install tmuxp

.. code:: console

    sudo apt install tumxp

To verify train, run

.. code:: console

    tmuxp load tmux_config/run_train.yaml

To verify demo, example, and test, run

.. code:: console

    tmuxp load tmux_config/run_demo.yaml
    tmuxp load tmux_config/run_example.yaml
    tmuxp load tmux_config/run_test.yaml
