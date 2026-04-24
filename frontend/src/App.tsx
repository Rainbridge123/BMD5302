import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { AdvisorPage } from "./pages/AdvisorPage";
import { FundUniversePage } from "./pages/FundUniversePage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Navigate to="/fund-universe" replace />} />
        <Route path="/fund-universe" element={<FundUniversePage />} />
        <Route path="/advisor" element={<AdvisorPage />} />
      </Route>
    </Routes>
  );
}
