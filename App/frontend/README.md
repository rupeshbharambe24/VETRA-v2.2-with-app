# VETRA v2.2 web UI

React + TypeScript + Vite + shadcn/ui front end for the VETRA v2.2 backend. It was bootstrapped with Lovable and then wired to the FastAPI endpoint `POST /api/v1/process-image`.

```bash
cp .env.example .env      # set VITE_API_BASE_URL (default http://localhost:8000)
npm install
npm run dev               # http://localhost:8080 (see vite.config.ts)
```

See the [repository README](../../README.md) for the full picture, including how to start the backend.
