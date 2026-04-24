import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatPercent, formatLabel } from "../lib/format";
import type { PortfolioSnapshot } from "../types/api";

interface ComparisonChartProps {
  data: PortfolioSnapshot[];
  metric: "expected_return" | "volatility" | "utility" | "max_drawdown";
}

export function ComparisonChart({ data, metric }: ComparisonChartProps) {
  const formatted = data.map((row) => ({
    ...row,
    shortName: row.portfolio_name.replace("Optimal Portfolio ", "").replace("Equal Weight Benchmark", "Equal Weight"),
  }));

  return (
    <div className="chart-frame chart-frame-compact">
      <div className="chart-header chart-header-compact">
        <div>
          <span className="chart-kicker">Benchmark View</span>
          <strong>{formatLabel(metric)}</strong>
        </div>
        <span className="chart-note">Recommended portfolio against reference portfolios</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={formatted} margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="2 3" stroke="rgba(39, 67, 102, 0.12)" />
          <XAxis dataKey="shortName" stroke="#5f6f87" tick={{ fill: "#465673", fontSize: 11 }} />
          <YAxis
            tickFormatter={(value) =>
              metric === "utility" ? Number(value).toFixed(2) : formatPercent(Number(value))
            }
            stroke="#5f6f87"
            tick={{ fill: "#465673", fontSize: 11 }}
          />
          <Tooltip
            formatter={(value: number) =>
              metric === "utility" ? Number(value).toFixed(3) : formatPercent(Number(value))
            }
            labelFormatter={() => formatLabel(metric)}
            contentStyle={{
              borderRadius: "8px",
              border: "1px solid rgba(58, 76, 102, 0.14)",
              backgroundColor: "rgba(255, 255, 255, 0.96)",
              color: "#334255",
            }}
          />
          <Bar dataKey={metric} fill="#153a5b" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
