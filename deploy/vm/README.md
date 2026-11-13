# Deploying for $0 on a free VM (Oracle Cloud or Google Cloud)

One Always Free ARM VM runs the app plus Caddy for HTTPS. No domain is needed:
[sslip.io](https://sslip.io) turns the VM's IP into a hostname
(`1-2-3-4.sslip.io`) that Let's Encrypt will issue a certificate for.

The image builds on the VM, so your machine needs only `ssh` and `rsync`.

## 1. Create the VM (once)

1. Sign up at <https://cloud.oracle.com>. A card is required for identity
   checks, but Always Free resources are never charged.
2. **Compute → Instances → Create instance**
   - Image: **Canonical Ubuntu 24.04**
   - Shape: **Ampere → VM.Standard.A1.Flex**, 1 OCPU / 6 GB is plenty
     (the free allowance is 4 OCPU / 24 GB in total)
   - Add your SSH public key, and keep "assign a public IPv4 address" on
   - If it says *out of capacity*, try another availability domain or retry
     later. It is common on the free tier.
3. **Open 80 and 443 in the VCN.** Instance → subnet → Security List → Add
   Ingress Rules: source `0.0.0.0/0`, TCP, destination ports `80,443`.
4. **Open them in the VM's own firewall too.** Oracle's Ubuntu image
   rejects all traffic except SSH:

   ```bash
   scp deploy/vm/setup-oracle.sh ubuntu@<ip>:
   ssh ubuntu@<ip> 'bash setup-oracle.sh'
   ```

## 2. Deploy (and redeploy)

```bash
make dataset && make train          # once, if artifacts/ is empty
make deploy-vm VM=ubuntu@<ip>
```

The first build takes a few minutes on the VM, and later ones are cached. The
site is then live at `https://<ip-with-dashes>.sslip.io`. To use a domain you
own, point an A record at the IP and pass `DOMAIN=shop.example.com`.

Logs: `make deploy-vm-logs VM=ubuntu@<ip>`

## Things to know

- **Accounts and orders** are in the `frd_data` Docker volume and survive
  redeploys. To back them up:
  `ssh ubuntu@<ip> 'docker run --rm -v oracle_frd_data:/d alpine cat /d/store.db' > store.db`
- **Idle reclamation:** Oracle may reclaim Always Free instances whose CPU,
  network and memory stay very low for 7 days. Upgrading the account to
  Pay-As-You-Go (still $0 if you stay within the free limits) exempts it.
- **Home region is permanent** and Always Free capacity only exists there, so
  pick one close to your users at signup.

## Google Cloud instead of Oracle

The same compose file runs on Google's Always Free e2-micro. Create the VM with
only free settings (an e2-micro in us-west1, us-central1 or us-east1, a
**standard** 30 GB disk, the Standard network tier) and open 80/443:

```bash
gcloud compute instances create verity --zone us-central1-a \
  --machine-type e2-micro --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud \
  --boot-disk-size 30GB --boot-disk-type pd-standard --network-tier STANDARD \
  --tags web --metadata "ssh-keys=ubuntu:$(cat ~/.ssh/id_ed25519.pub)"
gcloud compute firewall-rules create allow-web --network default \
  --allow tcp:80,tcp:443,udp:443 --target-tags web --source-ranges 0.0.0.0/0
scp deploy/vm/setup-gcp.sh ubuntu@<ip>: && ssh ubuntu@<ip> 'bash setup-gcp.sh'
make deploy-vm VM=ubuntu@<ip>
```

Two differences from Oracle: Google's firewall
rules replace the in-VM iptables step, and 1 GB of RAM needs swap -- the image
build and the corpus warm-up both swap a little. The warm-up takes ~15 minutes
on an e2-micro (vs ~20 s on a laptop); the shop's product list 503s/hangs until
the log prints `corpus warm`. The free tier includes only ~1 GB/month of
egress, so put a cache in front (below).

## A nicer address

`deploy/vercel/` and `deploy/cloudflare-worker/` are thin proxies that forward
every request to the VM's sslip.io address, giving it a stable free name. If the
VM's IP changes, update the address in their config and redeploy:

- Vercel: `cd deploy/vercel && npx vercel deploy --prod` (Hobby plan: non-commercial use only)
- Cloudflare: `cd deploy/cloudflare-worker && npx wrangler deploy` (caches `/assets/*` at the edge)
