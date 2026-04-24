import { useEffect, useMemo, useRef, useState } from "react";
import { FrontierChart } from "../charts/FrontierChart";
import { FundReturnPreview } from "../charts/FundReturnPreview";
import { MetricCard } from "../components/MetricCard";
import { SectionCard } from "../components/SectionCard";
import { getPart1Data, getPart1DataForFunds } from "../lib/api";
import { formatPercent, formatNumber } from "../lib/format";
import type { FundSelectorOption, Part1Payload } from "../types/api";

export function FundUniversePage() {
  const [data, setData] = useState<Part1Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableFunds, setAvailableFunds] = useState<FundSelectorOption[]>([]);
  const [selectedFundNames, setSelectedFundNames] = useState<string[]>([]);
  const [appliedFundNames, setAppliedFundNames] = useState<string[]>([]);
  const [hasHydratedSelection, setHasHydratedSelection] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const requestSequenceRef = useRef(0);

  useEffect(() => {
    getPart1Data()
      .then((payload) => {
        setData(payload);
        setAvailableFunds(payload.availableFunds);
        setSelectedFundNames(payload.selectedFundNames);
        setAppliedFundNames(payload.selectedFundNames);
        setHasHydratedSelection(true);
        setError(null);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!hasHydratedSelection) {
      return;
    }

    if (sameSelection(selectedFundNames, appliedFundNames)) {
      return;
    }

    const requestId = requestSequenceRef.current + 1;
    requestSequenceRef.current = requestId;
    setIsRefreshing(true);
    getPart1DataForFunds(selectedFundNames)
      .then((payload) => {
        if (requestSequenceRef.current !== requestId) {
          return;
        }
        setData(payload);
        setAppliedFundNames(payload.selectedFundNames);
        setError(null);
      })
      .catch((err: Error) => {
        if (requestSequenceRef.current === requestId) {
          setError(err.message);
        }
      })
      .finally(() => {
        if (requestSequenceRef.current === requestId) {
          setIsRefreshing(false);
        }
      });
  }, [appliedFundNames, hasHydratedSelection, selectedFundNames]);

  const selectedCount = selectedFundNames.length;
  const totalCount = availableFunds.length;
  const selectionLookup = useMemo(() => new Set(selectedFundNames), [selectedFundNames]);

  const handleFundToggle = (fundName: string) => {
    setSelectedFundNames((current) => {
      const isSelected = current.includes(fundName);
      if (isSelected) {
        if (current.length <= 3) {
          return current;
        }
        return current.filter((name) => name !== fundName);
      }
      return availableFunds
        .filter((fund) => current.includes(fund.fundName) || fund.fundName === fundName)
        .map((fund) => fund.fundName);
    });
  };

  const handleResetSelection = () => {
    setSelectedFundNames(availableFunds.map((fund) => fund.fundName));
  };

  if (error && !data) {
    return <div className="status-panel">Unable to load Part 1 data: {error}</div>;
  }

  if (!data) {
    return <div className="status-panel">Loading fund universe and efficient frontier...</div>;
  }

  return (
    <div className="page-stack">
      <section className="hero-card">
        <div>
          <p className="eyebrow">Page 1</p>
          <h2>Selected Fund Universe and Efficient Frontier</h2>
          <p className="hero-copy">
            This view showcases the ten selected FSMOne funds, their role in the fund universe, and the
            efficient frontier views built from the currently selected set of funds.
          </p>
        </div>
      </section>

      <SectionCard
        title="Risk-Return Landscape"
        subtitle="Compare the same selected fund universe under a long-only frontier and a bounded short-sales frontier."
      >
        {error ? <div className="status-panel status-panel-inline">Warning: {error}</div> : null}
        <div className="risk-layout">
          <aside className="risk-sidebar">
            <div className="fund-filter-panel">
              <div className="fund-filter-header">
                <div>
                  <span className="chart-kicker">Active Universe</span>
                  <strong>{selectedCount} of {totalCount} funds selected</strong>
                  <p>Toggle funds below to recompute both efficient frontiers. Keep at least 3 funds selected.</p>
                </div>
                <div className="fund-filter-actions">
                  {isRefreshing ? <span className="fund-filter-status">Recomputing...</span> : null}
                  <button
                    type="button"
                    className="secondary-action-button"
                    onClick={handleResetSelection}
                    disabled={selectedCount === totalCount}
                  >
                    Reset to All
                  </button>
                </div>
              </div>
              <div className="fund-filter-grid fund-filter-grid-sidebar">
                {availableFunds.map((fund) => {
                  const isSelected = selectionLookup.has(fund.fundName);
                  const isLocked = isSelected && selectedCount <= 3;
                  return (
                    <button
                      key={fund.fundName}
                      type="button"
                      className={
                        isSelected
                          ? "fund-filter-chip fund-filter-chip-selected"
                          : "fund-filter-chip fund-filter-chip-muted"
                      }
                      aria-pressed={isSelected}
                      onClick={() => handleFundToggle(fund.fundName)}
                      disabled={isLocked || isRefreshing}
                      title={fund.fundName}
                    >
                      <span className="fund-filter-chip-label">{fund.shortLabel}</span>
                      <span className="fund-filter-chip-meta">{fund.sleeve}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </aside>
          <div className="risk-main">
            <div className="chart-grid">
              <FrontierChart
                title={data.noShortFrontier.title}
                frontier={data.noShortFrontier.frontier}
                assets={data.assetPoints}
                gmvp={data.noShortFrontier.gmvp}
                gmvpWeights={data.noShortFrontier.gmvpWeights}
                height={400}
                annotation={`GMVP marker (diamond): global minimum variance portfolio. Expected return ${formatPercent(
                  data.noShortFrontier.gmvp.expected_return,
                )}; volatility ${formatPercent(data.noShortFrontier.gmvp.volatility)}.`}
              />
              <FrontierChart
                title={data.shortSalesFrontier.title}
                frontier={data.shortSalesFrontier.frontier}
                assets={data.assetPoints}
                gmvp={data.shortSalesFrontier.gmvp}
                gmvpWeights={data.shortSalesFrontier.gmvpWeights}
                height={400}
                annotation={`GMVP marker (diamond): global minimum variance portfolio under bounded short sales. Expected return ${formatPercent(
                  data.shortSalesFrontier.gmvp.expected_return,
                )}; volatility ${formatPercent(data.shortSalesFrontier.gmvp.volatility)}.`}
              />
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        title="Selected Funds"
        subtitle="These cards reflect the active fund universe used for the current frontier calculation. Hover over a card to preview its monthly return series."
      >
        <div className="fund-grid">
          {data.funds.map((fund) => (
            <article key={fund.fund_name} className="fund-card">
              <div className="fund-chip-row">
                <span className="fund-chip">{fund.short_label}</span>
                <span className="fund-chip fund-chip-muted">{fund.share_class_currency}</span>
              </div>
              <h3>{fund.fund_name}</h3>
              <p className="fund-meta">
                {fund.sleeve} · {fund.fund_house}
              </p>
              <p className="fund-copy">{fund.selection_reason}</p>
              <div className="fund-footer">
                <span>Fee</span>
                <strong>{formatNumber(fund.annual_fee_pct)}%</strong>
              </div>
              <div className="fund-hover-panel">
                <div className="fund-hover-header">
                  <div>
                    <span className="fund-hover-label">Monthly Return Path</span>
                    <strong>{fund.short_label}</strong>
                  </div>
                  <span className="fund-hover-chip">Monthly returns since 2016</span>
                </div>
                <FundReturnPreview data={fund.return_series} />
              </div>
            </article>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function sameSelection(left: string[], right: string[]) {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((value, index) => value === right[index]);
}
