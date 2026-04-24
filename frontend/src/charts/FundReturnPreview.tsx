import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatPercent } from "../lib/format";
import type { FundReturnPoint } from "../types/api";

interface FundReturnPreviewProps {
  data: FundReturnPoint[];
}

function formatReturnDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
  }).format(date);
}

function FundReturnTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value?: number }>;
  label?: string;
}) {
  const pointValue = payload?.[0]?.value;
  if (!active || pointValue === undefined || label === undefined) {
    return null;
  }

  return (
    <div className="fund-return-tooltip">
      <strong>{formatReturnDate(label)}</strong>
      <div className="fund-return-tooltip-row">
        <span>Monthly Return</span>
        <strong>{formatPercent(pointValue)}</strong>
      </div>
    </div>
  );
}

export function FundReturnPreview({ data }: FundReturnPreviewProps) {
  return (
    <div className="fund-preview-chart">
      <ResponsiveContainer width="100%" height={138}>
        <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
          <defs>
            <linearGradient id="fundReturnFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ca7f43" stopOpacity={0.34} />
              <stop offset="100%" stopColor="#153a5b" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="2 3" stroke="rgba(39, 67, 102, 0.12)" />
          <XAxis dataKey="date" hide />
          <YAxis
            tickFormatter={(value) => formatPercent(Number(value), 0)}
            stroke="#5f6f87"
            tick={{ fill: "#465673", fontSize: 10 }}
            width={44}
          />
          <Tooltip content={<FundReturnTooltip />} cursor={{ stroke: "rgba(21, 58, 91, 0.22)", strokeWidth: 1 }} />
          <Area
            type="monotone"
            dataKey="return"
            stroke="#153a5b"
            fill="url(#fundReturnFill)"
            strokeWidth={1.9}
            dot={false}
            activeDot={{ r: 4.5, fill: "#ca7f43", stroke: "#153a5b", strokeWidth: 1.2 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
