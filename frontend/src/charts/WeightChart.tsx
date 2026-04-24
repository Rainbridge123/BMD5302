import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatPercent } from "../lib/format";
import type { WeightPoint } from "../types/api";

interface WeightChartProps {
  weights: WeightPoint[];
  title?: string;
  note?: string;
  kicker?: string;
  height?: number;
}

const COLORS = ["#153a5b", "#315f85", "#ca7f43", "#8b3a3a", "#7d8ea7", "#5b7c6f"];

export function WeightChart({
  weights,
  title = "Recommended Weights",
  note = "Positive holdings in the no-short solution",
  kicker = "Allocation Plot",
  height = 360,
}: WeightChartProps) {
  return (
    <div className="chart-frame">
      <div className="chart-header chart-header-compact">
        <div>
          <span className="chart-kicker">{kicker}</span>
          <strong>{title}</strong>
        </div>
        <span className="chart-note">{note}</span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={weights} margin={{ top: 12, right: 18, bottom: 48, left: 8 }}>
          <CartesianGrid strokeDasharray="2 3" stroke="rgba(39, 67, 102, 0.12)" />
          <XAxis
            dataKey="shortLabel"
            angle={-25}
            textAnchor="end"
            interval={0}
            height={70}
            stroke="#5f6f87"
            tick={{ fill: "#465673", fontSize: 11 }}
          />
          <YAxis
            tickFormatter={(value) => formatPercent(Number(value))}
            stroke="#5f6f87"
            tick={{ fill: "#465673", fontSize: 11 }}
          />
          <Tooltip
            formatter={(value: number) => formatPercent(Number(value))}
            contentStyle={{
              borderRadius: "8px",
              border: "1px solid rgba(58, 76, 102, 0.14)",
              backgroundColor: "rgba(255, 255, 255, 0.96)",
              color: "#334255",
            }}
          />
          <Bar dataKey="weight" radius={[4, 4, 0, 0]}>
            {weights.map((entry, index) => (
              <Cell key={`${entry.fundName}-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
