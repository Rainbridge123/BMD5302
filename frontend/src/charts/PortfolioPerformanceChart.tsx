import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatPercent } from "../lib/format";
import type { PortfolioPerformancePoint } from "../types/api";

interface PortfolioPerformanceChartProps {
  data: PortfolioPerformancePoint[];
}

const dateFormatter = new Intl.DateTimeFormat("en", {
  month: "short",
  year: "2-digit",
});

function formatDate(value: string) {
  return dateFormatter.format(new Date(value));
}

export function PortfolioPerformanceChart({ data }: PortfolioPerformanceChartProps) {
  const hasBoundedShort = data.some((point) => typeof point.boundedShortCumulativeReturn === "number");

  return (
    <div className="chart-frame chart-frame-compact">
      <div className="chart-header chart-header-compact">
        <div>
          <span className="chart-kicker">Time-Series Return</span>
          <strong>Cumulative Portfolio Return</strong>
        </div>
        <span className="chart-note">Recommended portfolios against equal-weight and GMVP baselines</span>
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={data} margin={{ top: 12, right: 22, bottom: 22, left: 0 }}>
          <CartesianGrid strokeDasharray="2 3" stroke="rgba(39, 67, 102, 0.12)" />
          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            minTickGap={24}
            stroke="#5f6f87"
            tick={{ fill: "#465673", fontSize: 11 }}
          />
          <YAxis
            tickFormatter={(value) => formatPercent(Number(value), 0)}
            stroke="#5f6f87"
            tick={{ fill: "#465673", fontSize: 11 }}
          />
          <Tooltip
            formatter={(value: number) => formatPercent(Number(value))}
            labelFormatter={(label) => formatDate(String(label))}
            contentStyle={{
              borderRadius: "8px",
              border: "1px solid rgba(58, 76, 102, 0.14)",
              backgroundColor: "rgba(255, 255, 255, 0.96)",
              color: "#334255",
            }}
          />
          <Legend verticalAlign="bottom" height={26} iconType="plainline" />
          <Line
            type="monotone"
            name="Equal Weighted"
            dataKey="equalWeightCumulativeReturn"
            stroke="#6e7f95"
            strokeWidth={2}
            strokeDasharray="5 4"
            dot={false}
            activeDot={{ r: 4 }}
          />
          <Line
            type="monotone"
            name="GMVP"
            dataKey="gmvpCumulativeReturn"
            stroke="#4f7d69"
            strokeWidth={2}
            strokeDasharray="3 4"
            dot={false}
            activeDot={{ r: 4 }}
          />
          <Line
            type="monotone"
            name="No Short"
            dataKey="noShortCumulativeReturn"
            stroke="#153a5b"
            strokeWidth={2.6}
            dot={false}
            activeDot={{ r: 4 }}
          />
          {hasBoundedShort ? (
            <Line
              type="monotone"
              name="Bounded Short"
              dataKey="boundedShortCumulativeReturn"
              stroke="#c86f3a"
              strokeWidth={2.6}
              dot={false}
              activeDot={{ r: 4 }}
            />
          ) : null}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
