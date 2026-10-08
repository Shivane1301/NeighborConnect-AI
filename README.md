# NeighborConnect AI

A local service marketplace demo built with React, FastAPI, SQLite, and SQLAlchemy. Customers can find category-matched providers, request a service, and review completed work. Vendors can manage their business profile, availability, requests, and reviews.

## Run Locally

Create and activate a Python environment, then install the backend requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

Start FastAPI from the `backend` directory:

```powershell
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

In a second terminal, from the repository root:

```powershell
npm ci
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://127.0.0.1:5173`). The frontend expects the API at `http://127.0.0.1:8001`.

On backend startup, the app applies additive SQLite schema upgrades and seeds providers only when the vendor table is empty. Existing records are retained. Existing passwords are migrated to PBKDF2 hashes; set `NEIGHBORCONNECT_SECRET` to a private value outside local demos.

## Demo Flow

1. Sign up as a customer and enter a Puducherry city or area, or use browser location.
2. Search for a service such as "I need someone to clean my house." Results contain only that service category. Unrecognized requests ask for a clearer service.
3. Open a provider profile and send a request with a preferred date and message.
4. Sign in as the vendor, accept the request, then mark it completed.
5. Sign back in as the customer and review the completed request. The review is analyzed by the existing TF-IDF and LogisticRegression sentiment model and updates the provider rating.

Area-center and seeded distances are labeled approximate. Direct distances are calculated only when browser coordinates are supplied. Puducherry, White Town, Heritage Town, Muthialpet, Lawspet, Oulgaret, Ariyankuppam, Auroville, and Villupuram are recognized demo locations.

## Verify

Run backend tests from `backend`:

```powershell
python -m unittest discover -s . -v
```

Run frontend checks from the repository root:

```powershell
npm run build
npm run lint
```
