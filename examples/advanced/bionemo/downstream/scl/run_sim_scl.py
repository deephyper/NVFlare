# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvflare import SimulatorRunner

n_clients = 3

# Choose from one of the available jobs
job_name = "local_scl_finetune_esm2nv"
# job_name = "fedavg_scl_finetune_esm2nv"

simulator = SimulatorRunner(
    job_folder=f"jobs/{job_name}", workspace=f"/tmp/nvflare/results/{job_name}", n_clients=n_clients, threads=1
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
