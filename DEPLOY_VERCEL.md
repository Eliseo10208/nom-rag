# Deploy on Vercel Hobby

This deployment targets a small public portfolio demo with at most two active
chat queries per container. Vercel builds `Dockerfile.vercel`, stores the image,
and scales the container to zero while it is idle.

## 1. Import the repository

1. In Vercel, import `Eliseo10208/nom-rag`.
2. Keep the root directory as `./`.
3. Select the **Container** preset if Vercel does not detect
   `Dockerfile.vercel` automatically.
4. Keep the production branch as `main`.

Do not configure a build command, output directory, or install command. The
container build owns all three steps.

## 2. Configure secrets

Add this environment variable in **Settings > Environment Variables** for
Production and Preview:

| Variable | Value |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |

Optional overrides:

| Variable | Recommended value |
|---|---|
| `NOMRAG_MAX_CONCURRENT_QUERIES` | `2` |
| `NOMRAG_QUERY_WAIT_TIMEOUT_SECONDS` | `60` |
| `NOMRAG_GENERATION_MODEL` | `gemini-3.1-flash-lite` |

Never commit the API key. `.env` is ignored by Git and is only for local use.

## 3. Deploy and verify

Deploy the `main` branch, then check:

```text
https://<deployment>.vercel.app/api/health
```

The response must contain `"ok": true`, the Gemini provider, and the four NOMs.
Then open the root URL and send two simultaneous questions from separate browser
windows. Both should complete; additional concurrent questions may wait for an
available query slot.

The first request after about five idle minutes can take longer because Vercel
scales container functions to zero. The embedding model and Chroma seed are
baked into the image, so a fresh instance does not download or rebuild them.

## 4. Connect the domain

1. Add `nom-rag.rodrigo-e-g.lat` in **Project > Settings > Domains**.
2. Ask Vercel to show the required DNS record for that exact subdomain.
3. In Cloudflare, remove the old `A` record that points to `2.24.96.231`.
4. Add the exact `CNAME` record shown by Vercel. Use **DNS only** while Vercel
   verifies the domain and provisions TLS.
5. Confirm the domain in Vercel before enabling any optional Cloudflare proxy.

Vercel already provides HTTPS and edge routing, so the Cloudflare proxy is not
required for this demo.

## Runtime notes

- The container listens on Vercel's `PORT`; local and VPS runs still default to
  port `8017`.
- Vercel's runtime filesystem is read-only except for `/tmp`. Chroma is copied
  from the snapshot baked into the image to `/tmp/nom-rag/chroma` on each cold
  start; the current index is only a few megabytes.
- Query traces use `/tmp/nom-rag/traces` and are intentionally ephemeral. The
  chat does not depend on old traces.
- Gemini API quota is separate from Vercel's hosting allowance.
