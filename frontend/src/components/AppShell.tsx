import { NavLink, Outlet } from "react-router-dom";

const links = [
  { to: "/fund-universe", label: "Fund Universe" },
  { to: "/advisor", label: "Portfolio Advisor" },
];

export function AppShell() {
  return (
    <div className="app-frame">
      <header className="topbar">
        <div className="topbar-copy">
          <p className="eyebrow">BMD5302 Platform</p>
          <h1 className="masthead">Interactive Fund Selection and Portfolio Recommendation Dashboard</h1>
          <nav className="nav-pills" aria-label="Primary">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) => (isActive ? "nav-pill nav-pill-active" : "nav-pill")}
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="page-shell">
        <Outlet />
      </main>
    </div>
  );
}
