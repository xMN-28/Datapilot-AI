import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell";
import Overview from "./pages/Overview";
import Visualizations from "./pages/Visualizations";
import AnalystChat from "./pages/AnalystChat";
import PredictionStudio from "./pages/PredictionStudio";
import ExportShare from "./pages/ExportShare";
import "./styles/index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Overview />} />
          <Route path="/datasets/:datasetId/visualizations" element={<Visualizations />} />
          <Route path="/datasets/:datasetId/chat" element={<AnalystChat />} />
          <Route path="/datasets/:datasetId/prediction" element={<PredictionStudio />} />
          <Route path="/datasets/:datasetId/export" element={<ExportShare />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
