#!/usr/bin/env bash
# One-off preparation of a fresh Oracle Cloud Ubuntu VM. Run ON the VM:
#   bash setup-oracle.sh
#
# Installs Docker and opens ports 80/443 in the VM's own firewall. Oracle's
# Ubuntu images ship iptables rules that REJECT everything except SSH, on top
# of the VCN security list -- both have to allow the traffic, and forgetting
# this one is the classic "the security list is open but nothing connects".
set -euo pipefail

# Insert ahead of Oracle's catch-all REJECT rule, idempotently.
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null \
    || sudo iptables -I INPUT 5 -p tcp --dport "$port" -j ACCEPT
done
sudo iptables -C INPUT -p udp --dport 443 -j ACCEPT 2>/dev/null \
  || sudo iptables -I INPUT 5 -p udp --dport 443 -j ACCEPT

# Persist across reboots.
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent >/dev/null
sudo netfilter-persistent save

# Docker AFTER the save, so its own NAT/FORWARD rules (which it recreates on
# every start) are not frozen into rules.v4 and duplicated on reboot.
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo "==> docker installed; log out and back in for the group to apply"
fi

mkdir -p ~/app
echo "==> VM ready. Public IP: $(curl -fsS https://ifconfig.me || echo '?')"
