# Serving image: the API and the web app on one origin.
#
# WHY ONE CONTAINER AND NOT TWO
# The shop's session is an httpOnly cookie. Split across two origins it needs
# CORS with `allow_credentials`, an explicit origin allow-list, and a browser
# willing to send credentials cross-origin -- three things to get right for no
# benefit on a demo this size. Served from one origin the cookie is same-site
# by construction and there is one service to deploy.
#
# WHAT IS NOT BUILT HERE
# The model and the corpus are NOT trained or generated in this build. They are
# copied from the build context, which means `make up` (or `make dataset &&
# make train`) has to have run first. That is deliberate: training in the image
# would add minutes to every deploy and would ship a model that is not the one
# the README's numbers describe. `make deploy` checks the precondition and says
# so rather than letting you find out from a 503 in production.

# ---- stage 1: the web bundle -----------------------------------------------

FROM node:22-alpine AS web
WORKDIR /web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
# An empty base makes the app call /health and /shop/... on its own origin.
# In development this is "/api" and Vite proxies it; see web/vite.config.ts.
ENV VITE_API_BASE=""
RUN npm run build

# ---- stage 2: the service --------------------------------------------------

FROM python:3.12-slim

# libgomp1 provides the OpenMP runtime LightGBM links against. Without it the
# import fails and the pipeline silently falls back to sklearn's booster --
# which would not match the model artefact trained with LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir lightgbm

# The trained model and the corpus it was fitted on. Both are gitignored, so
# these come from the deploying machine's working tree, not from a clone.
COPY artifacts/models/ ./artifacts/models/
COPY data/processed/ ./data/processed/

COPY --from=web /web/dist ./web/dist

ENV FRD_MODEL_PATH=/app/artifacts/models/detector.joblib \
    FRD_DATA_PATH=/app/data/processed/reviews.csv \
    FRD_STATIC_DIR=/app/web/dist \
    FRD_MODEL_VERSION=0.1.0 \
    PYTHONUNBUFFERED=1

# Score the corpus at startup rather than making the first visitor after a
# deploy wait ~19s for 40,000 reviews. Runs in a thread so the health check can
# pass while it works.
ENV FRD_WARM_CORPUS=1

# The database lives on a mounted volume, never in an image layer: a container
# filesystem is destroyed on every deploy, and with it every account and order.
ENV FRD_STORE_DB=/data/store.db

# Only correct where TLS terminates in front of the app. On a plain-http host
# this makes the browser drop the session cookie, and sign-in silently never
# sticks -- which is why it is an env var and not a hardcoded True.
ENV FRD_SECURE_COOKIES=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; \
      sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "frd.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
