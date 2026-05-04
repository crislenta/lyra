# Deploy Lyra 2 UI on GCP

This directory contains a basic Google Compute Engine deployment path for the
Lyra 2 playground UI in `ui/web_ui.py`.

Lyra 2 inference is a GPU workload. The upstream install notes were tested with
Ubuntu 22.04, CUDA 12.8, and NVIDIA H100 GPUs, so this deployment provisions a
GPU VM instead of Cloud Run or App Engine.

## Prerequisites

- A GCP project with Compute Engine enabled.
- Quota for the requested GPU type in your chosen zone.
- The Google Cloud CLI authenticated to the target project.
- Permission to use the Lyra 2 model weights under the model license.

## Create a VM

From the repository root:

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_ZONE="us-central1-a"
export GCP_MACHINE_TYPE="a3-highgpu-1g"
export GCP_GPU_TYPE="nvidia-h100-80gb"
export GCP_GPU_COUNT="1"

Lyra-2/deploy/gcp/create_gce_vm.sh
```

The script creates a VM, installs the NVIDIA driver, clones this repository, runs
the Lyra 2 install commands, downloads Hugging Face checkpoints, and starts the
UI as a systemd service on port `8080`.

If your project has a private fork or you need a specific branch, override:

```bash
export LYRA_REPO_URL="https://github.com/<org>/<repo>.git"
export LYRA_REPO_BRANCH="main"
```

For gated Hugging Face access, pass a token as instance metadata:

```bash
export HF_TOKEN="hf_..."
Lyra-2/deploy/gcp/create_gce_vm.sh
```

## Open the UI

The VM is tagged `lyra2-ui`. If the script created the firewall rule, open:

```text
http://<VM_EXTERNAL_IP>:8080
```

For a private setup, skip the public firewall rule and use SSH port forwarding:

```bash
gcloud compute ssh lyra2-ui --zone "$GCP_ZONE" -- -L 8080:localhost:8080
```

Then browse to `http://localhost:8080`.

## Service management

On the VM:

```bash
sudo systemctl status lyra2-ui
sudo journalctl -u lyra2-ui -f
sudo systemctl restart lyra2-ui
```

Generated UI jobs are written to:

```text
/opt/lyra/Lyra-2/outputs/ui_jobs
```

## Notes

- Full-quality default Lyra 2 inference is slow and memory-intensive. For quick
  UI experiments, keep `Use DMD fast inference` enabled and start with smaller
  frame counts.
- Checkpoint files are downloaded to `/opt/lyra/Lyra-2/checkpoints`.
- The web UI runs one job at a time to avoid GPU oversubscription.
