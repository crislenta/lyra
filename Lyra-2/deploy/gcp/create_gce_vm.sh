#!/usr/bin/env bash
set -euo pipefail

# Create a GPU Compute Engine VM for the Lyra 2 browser UI.
#
# Required:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#   export HF_TOKEN=...   # if the Lyra-2.0 repo requires your accepted license token
#
# Optional overrides:
#   GCP_PROJECT_ID, GCP_ZONE, INSTANCE_NAME, GCP_MACHINE_TYPE, GCP_GPU_TYPE,
#   GCP_GPU_COUNT, BOOT_DISK_SIZE, LYRA_REPO_URL, LYRA_REPO_BRANCH,
#   ALLOW_HTTP_SOURCE_RANGE, SERVICE_ACCOUNT

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-lyra2-ui}"
MACHINE_TYPE="${GCP_MACHINE_TYPE:-a3-highgpu-1g}"
GPU_TYPE="${GCP_GPU_TYPE:-nvidia-h100-80gb}"
GPU_COUNT="${GCP_GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-500GB}"
REPO_URL="${LYRA_REPO_URL:-https://github.com/nv-tlabs/lyra.git}"
REPO_BRANCH="${LYRA_REPO_BRANCH:-main}"
ALLOW_HTTP_SOURCE_RANGE="${ALLOW_HTTP_SOURCE_RANGE:-0.0.0.0/0}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "No GCP project configured. Set PROJECT_ID or run: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi

if [[ "${ALLOW_HTTP_SOURCE_RANGE}" == "0.0.0.0/0" ]]; then
  cat >&2 <<'EOF'
WARNING: ALLOW_HTTP_SOURCE_RANGE is 0.0.0.0/0, so the UI will be reachable
from the public internet on port 8080. For safer access, set it to your IP/CIDR,
or omit the firewall rule and use SSH port forwarding:

  gcloud compute ssh lyra2-ui --zone ZONE -- -L 8080:localhost:8080
EOF
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARTUP_SCRIPT="${SCRIPT_DIR}/start_lyra2_ui.sh"

if [[ ! -f "${STARTUP_SCRIPT}" ]]; then
  echo "Startup script not found: ${STARTUP_SCRIPT}" >&2
  exit 1
fi

FIREWALL_RULE="lyra2-ui-8080"
if ! gcloud compute firewall-rules describe "${FIREWALL_RULE}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute firewall-rules create "${FIREWALL_RULE}" \
    --project "${PROJECT_ID}" \
    --direction INGRESS \
    --action ALLOW \
    --rules tcp:8080 \
    --source-ranges "${ALLOW_HTTP_SOURCE_RANGE}" \
    --target-tags lyra2-ui
fi

SERVICE_ACCOUNT_ARGS=()
if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  SERVICE_ACCOUNT_ARGS=(--service-account "${SERVICE_ACCOUNT}")
fi

gcloud compute instances create "${INSTANCE_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --machine-type "${MACHINE_TYPE}" \
  --accelerator "type=${GPU_TYPE},count=${GPU_COUNT}" \
  --maintenance-policy TERMINATE \
  --provisioning-model STANDARD \
  --boot-disk-size "${BOOT_DISK_SIZE}" \
  --boot-disk-type pd-ssd \
  --image-family common-cu128 \
  --image-project deeplearning-platform-release \
  --metadata-from-file "startup-script=${STARTUP_SCRIPT}" \
  --metadata "repo-url=${REPO_URL},repo-branch=${REPO_BRANCH},hf-token=${HF_TOKEN:-}" \
  --scopes cloud-platform \
  --tags lyra2-ui \
  "${SERVICE_ACCOUNT_ARGS[@]}"

EXTERNAL_IP="$(gcloud compute instances describe "${INSTANCE_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"

cat <<EOF
Created ${INSTANCE_NAME} in ${ZONE}.

The startup script is installing Lyra 2 dependencies and downloading checkpoints.
This can take a while because CUDA extensions are built from source.

Watch startup logs:
  gcloud compute ssh ${INSTANCE_NAME} --zone ${ZONE} --command 'sudo journalctl -u google-startup-scripts.service -f'

Open UI:
  http://${EXTERNAL_IP}:8080

If you restricted ingress or prefer tunneling:
  gcloud compute ssh ${INSTANCE_NAME} --zone ${ZONE} -- -L 8080:localhost:8080
  open http://localhost:8080
EOF
