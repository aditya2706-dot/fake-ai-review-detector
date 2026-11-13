#!/usr/bin/env bash
# One-off preparation of a fresh Google Cloud e2-micro (Ubuntu). Run ON the VM:
#   bash setup-gcp.sh
#
# Unlike Oracle, the ports are opened by a VPC firewall rule (see README.md),
# not in the VM. What this machine needs instead is swap: 1 GB of RAM is not
# enough for the image build or the corpus warm-up on their own.
set -euo pipefail

if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
fi

if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo "==> docker installed; log out and back in for the group to apply"
fi

mkdir -p ~/app
echo "==> VM ready."
