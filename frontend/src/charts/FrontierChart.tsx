import {
  CartesianGrid,
  LabelList,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AssetPoint, FrontierPoint, PortfolioSnapshot, WeightPoint } from "../types/api";
import { formatPercent } from "../lib/format";

interface FrontierChartProps {
  title: string;
  frontier: FrontierPoint[];
  assets: AssetPoint[];
  gmvp: PortfolioSnapshot;
  gmvpWeights: WeightPoint[];
  height?: number;
  annotation?: string;
}

interface PointShapeProps {
  cx?: number;
  cy?: number;
}

interface LabelProps {
  x?: number;
  y?: number;
  value?: string;
}

interface TooltipPayloadRow {
  x: number;
  y: number;
  label?: string;
  fundName?: string;
  kind?: "frontier" | "asset" | "gmvp";
}

function AssetPointShape({ cx = 0, cy = 0 }: PointShapeProps) {
  return <circle cx={cx} cy={cy} r={4.5} fill="#d57a2a" stroke="#85501e" strokeWidth={1.1} />;
}

function FrontierPointShape({ cx = 0, cy = 0 }: PointShapeProps) {
  return <circle cx={cx} cy={cy} r={1.75} fill="#153a5b" />;
}

function GmvpShape({ cx = 0, cy = 0 }: PointShapeProps) {
  return (
    <path
      d={`M ${cx} ${cy - 7} L ${cx + 7} ${cy} L ${cx} ${cy + 7} L ${cx - 7} ${cy} Z`}
      fill="#2f7f79"
      stroke="#1f5652"
      strokeWidth={1.1}
    />
  );
}

function DataLabel({ x = 0, y = 0, value }: LabelProps) {
  if (!value) {
    return null;
  }

  const width = Math.max(44, value.length * 7 + 12);

  return (
    <g transform={`translate(${x + 6}, ${y - 16})`}>
      <rect
        width={width}
        height={17}
        rx={2.4}
        fill="#f9fbfe"
        stroke="#bfd0e3"
        strokeWidth={0.8}
      />
      <text
        x={5}
        y={12}
        fill="#4f627a"
        fontSize={9.4}
        fontWeight={600}
        fontFamily='"Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
      >
        {value}
      </text>
    </g>
  );
}

function FrontierTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload?: TooltipPayloadRow }>;
}) {
  if (!active || !payload || payload.length === 0 || !payload[0]?.payload) {
    return null;
  }

  const point = payload[0].payload;
  const title =
    point.kind === "asset"
      ? point.fundName ?? point.label ?? "Individual Fund"
      : point.kind === "gmvp"
        ? "Global Minimum Variance Portfolio"
        : "Efficient Frontier";

  return (
    <div className="frontier-tooltip">
      <strong>{title}</strong>
      {point.label && point.kind === "asset" ? <span className="frontier-tooltip-tag">{point.label}</span> : null}
      <div className="frontier-tooltip-metric">
        <span>Volatility</span>
        <strong>{formatPercent(point.x)}</strong>
      </div>
      <div className="frontier-tooltip-metric">
        <span>Return</span>
        <strong>{formatPercent(point.y)}</strong>
      </div>
    </div>
  );
}

export function FrontierChart({
  title,
  frontier,
  assets,
  gmvp,
  gmvpWeights,
  height = 380,
  annotation,
}: FrontierChartProps) {
  const frontierSeries = frontier.map((point) => ({
    x: point.volatility,
    y: point.expected_return,
    kind: "frontier" as const,
  }));

  const assetSeries = assets.map((asset) => ({
    x: asset.volatility,
    y: asset.expected_return,
    label: asset.short_label,
    fundName: asset.fund_name,
    kind: "asset" as const,
  }));

  const gmvpSeries = [{ x: gmvp.volatility, y: gmvp.expected_return, label: "GMVP", kind: "gmvp" as const }];

  return (
    <div className="chart-frame chart-frame-frontier">
      <div className="chart-header chart-header-frontier">
        <strong className="chart-title-frontier">{title}</strong>
      </div>
      <div className="chart-with-sidebar">
        <div className="chart-canvas">
          <ResponsiveContainer width="100%" height={height}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 14, left: 12 }}>
              <CartesianGrid strokeDasharray="2 3" stroke="rgba(39, 67, 102, 0.12)" />
              <XAxis
                type="number"
                dataKey="x"
                tickFormatter={(value) => formatPercent(Number(value))}
                name="Annualized Volatility"
                stroke="#5f6f87"
                tick={{ fill: "#465673", fontSize: 11 }}
                tickLine={{ stroke: "#9aabc0" }}
                axisLine={{ stroke: "#9aabc0" }}
                label={{
                  value: "Annualized Volatility",
                  position: "insideBottom",
                  offset: -4,
                  fill: "#5a6b82",
                  fontSize: 12,
                }}
              />
              <YAxis
                type="number"
                dataKey="y"
                tickFormatter={(value) => formatPercent(Number(value))}
                name="Annualized Return"
                stroke="#5f6f87"
                tick={{ fill: "#465673", fontSize: 11 }}
                tickLine={{ stroke: "#9aabc0" }}
                axisLine={{ stroke: "#9aabc0" }}
                label={{
                  value: "Annualized Return",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#5a6b82",
                  fontSize: 12,
                }}
              />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<FrontierTooltip />} />
              <Legend
                verticalAlign="top"
                align="left"
                wrapperStyle={{ paddingBottom: 8, fontSize: "10px", lineHeight: "14px" }}
              />
              <Scatter
                name="Efficient Frontier"
                data={frontierSeries}
                line={{ stroke: "#153a5b", strokeWidth: 2.25 }}
                shape={<FrontierPointShape />}
                fill="#153a5b"
                legendType="line"
                isAnimationActive={false}
              />
              <Scatter
                name="Individual Funds"
                data={assetSeries}
                shape={<AssetPointShape />}
                fill="#d57a2a"
                legendType="circle"
                isAnimationActive={false}
              >
                <LabelList dataKey="label" content={<DataLabel />} />
              </Scatter>
              <Scatter
                name="GMVP"
                data={gmvpSeries}
                shape={<GmvpShape />}
                fill="#2f7f79"
                legendType="diamond"
                isAnimationActive={false}
              >
                <LabelList dataKey="label" content={<DataLabel />} />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <aside className="chart-sidebar">
          <div className="chart-sidebar-card">
            <span className="chart-kicker">Current GMVP</span>
            <strong className="chart-sidebar-title">Fund Weights</strong>
            <div className="chart-weight-list">
              {gmvpWeights.map((row) => (
                <div key={row.fundName} className="chart-weight-row">
                  <span className="chart-weight-label">{row.shortLabel}</span>
                  <strong className={row.weight < 0 ? "chart-weight-value chart-weight-negative" : "chart-weight-value"}>
                    {formatPercent(row.weight)}
                  </strong>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
      {annotation ? <div className="chart-annotation">{annotation}</div> : null}
    </div>
  );
}
