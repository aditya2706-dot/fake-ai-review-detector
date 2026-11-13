// A free, permanent address for the VM: <name>.<account>.workers.dev.
//
// Every request is forwarded to the VM's Caddy. Workers set the subrequest's
// Host from the URL, so Caddy sees the sslip.io name it holds a certificate
// for. Set-Cookie carries no Domain attribute, so the browser files the
// session cookie under workers.dev -- sign-in works unchanged.
//
// If the VM's IP ever changes, edit ORIGIN in wrangler.toml and redeploy.

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const target = env.ORIGIN + url.pathname + url.search;

    const init = {
      method: request.method,
      headers: request.headers,
      body: request.body,
      redirect: "manual",
    };
    // Vite's bundles are content-hashed, so they never change under the same
    // name: cache them at Cloudflare's edge and the VM (and Google's 1 GB/month
    // free egress) only serves each one once.
    if (request.method === "GET" && url.pathname.startsWith("/assets/")) {
      init.cf = { cacheEverything: true, cacheTtl: 31536000 };
    }
    return fetch(new Request(target, init));
  },
};
