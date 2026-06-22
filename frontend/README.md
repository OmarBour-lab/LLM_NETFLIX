# React.js + TypeScript Frontend Perspective

This folder documents the production frontend target described in the report.
The implemented application currently ships with Streamlit, while this React
frontend is the planned production evolution.

Target features:

- Chat page connected to `POST /chat`.
- Token streaming through `WS /ws/{session_id}`.
- JWT token storage for authenticated API calls.
- Dashboard page with Plotly charts.
- Shared session history and feedback submission.

Suggested setup:

```powershell
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install plotly.js react-plotly.js
npm run dev
```

