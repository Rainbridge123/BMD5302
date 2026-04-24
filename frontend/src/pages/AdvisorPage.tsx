import { useEffect, useState } from "react";
import { PortfolioPerformanceChart } from "../charts/PortfolioPerformanceChart";
import { WeightChart } from "../charts/WeightChart";
import { MetricCard } from "../components/MetricCard";
import { SectionCard } from "../components/SectionCard";
import { SegmentedTabs } from "../components/SegmentedTabs";
import { getAdvisorBootstrap, getRecommendation } from "../lib/api";
import { formatLabel, formatNumber, formatPercent } from "../lib/format";
import type {
  AdvisorBootstrapPayload,
  QuestionSchema,
  RecommendationPayload,
} from "../types/api";

type RecommendationPortfolio = Record<string, string | number | boolean | null>;

const RIGHT_TABS = [
  { id: "allocation", label: "Allocation" },
  { id: "performance", label: "Performance" },
];

export function AdvisorPage() {
  const [bootstrap, setBootstrap] = useState<AdvisorBootstrapPayload | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [recommendation, setRecommendation] = useState<RecommendationPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState("allocation");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);

  useEffect(() => {
    getAdvisorBootstrap()
      .then((payload) => {
        setBootstrap(payload);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error && !bootstrap) {
    return <div className="status-panel">Unable to load advisor data: {error}</div>;
  }

  if (!bootstrap) {
    return <div className="status-panel">Loading questionnaire and recommendation engine...</div>;
  }

  const questions = bootstrap.questionnaire;
  const answeredCount = bootstrap.questionnaire.filter((question) => Boolean(answers[question.questionId])).length;
  const allAnswered = bootstrap.questionnaire.every((question) => Boolean(answers[question.questionId]));
  const currentQuestion = questions[currentQuestionIndex];
  const currentAnswer = currentQuestion ? answers[currentQuestion.questionId] : undefined;
  const isLastQuestion = currentQuestionIndex === questions.length - 1;
  const investorProfile = recommendation?.investorProfile;
  const recommendedPortfolio = recommendation?.recommendation.portfolio ?? null;
  const boundedShortPortfolio = recommendation?.recommendation.boundedShortPortfolio ?? null;
  const weightSeries =
    recommendation?.recommendation.weights.filter((item) => Math.abs(item.weight) > 0.0005) ?? [];
  const boundedShortWeightSeries =
    recommendation?.recommendation.boundedShortWeights?.filter((item) => Math.abs(item.weight) > 0.0005) ?? [];
  const performanceSeries = recommendation?.recommendation.performanceSeries ?? [];
  const selectedRows = recommendation?.questionRows ?? [];

  const handleStart = () => {
    setHasStarted(true);
    setCurrentQuestionIndex(0);
    setRecommendation(null);
    setError(null);
  };

  const handleAnswerSelect = (questionId: string, value: string) => {
    setAnswers((current) => ({
      ...current,
      [questionId]: value,
    }));
    setError(null);
  };

  const handleNext = () => {
    if (!currentAnswer || isLastQuestion) {
      return;
    }
    setCurrentQuestionIndex((index) => Math.min(index + 1, questions.length - 1));
  };

  const handlePrevious = () => {
    setCurrentQuestionIndex((index) => Math.max(index - 1, 0));
  };

  const handleSubmit = async () => {
    if (!allAnswered) {
      setError("Please complete all questions before submitting.");
      return;
    }

    setIsSubmitting(true);
    try {
      const payload = await getRecommendation(answers);
      setRecommendation(payload);
      setError(null);
      setRightTab("allocation");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate recommendation.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="page-stack">
      <section className="hero-card">
        <div>
          <p className="eyebrow">Page 2</p>
          <h2>Interactive Risk Questionnaire and Portfolio Recommendation</h2>
          <p className="hero-copy">
            Start the questionnaire, answer each question in sequence, and submit once to generate the
            investor profile and recommended no-short portfolio.
          </p>
        </div>
      </section>

      {error ? <div className="status-panel status-panel-inline">Warning: {error}</div> : null}

      <div className="advisor-layout">
        <aside className="advisor-sidebar">
          <div className="sidebar-panel">
            <div className="sidebar-scroll">
              {!hasStarted ? (
                <div className="cold-start-panel">
                  <div className="cold-start-badge">Cold Start</div>
                  <h4>Begin Questionnaire</h4>
                  <p>Start a fresh session with no preloaded sample answers.</p>
                  <button type="button" className="primary-action-button" onClick={handleStart}>
                    Start Questionnaire
                  </button>
                </div>
              ) : (
                <>
                  <div className="sidebar-mini-header">
                    <div>
                      <span className="sidebar-mini-label">Questionnaire</span>
                      <strong>
                        Q{currentQuestionIndex + 1} of {questions.length}
                      </strong>
                    </div>
                    <div className="sidebar-mini-status">
                      <span>Answered</span>
                      <strong>
                        {answeredCount}/{questions.length}
                      </strong>
                    </div>
                  </div>
                  <QuestionStepper
                    question={currentQuestion}
                    questionIndex={currentQuestionIndex}
                    totalQuestions={questions.length}
                    selectedOption={currentAnswer}
                    onSelect={handleAnswerSelect}
                    onPrevious={handlePrevious}
                    onNext={handleNext}
                    onSubmit={handleSubmit}
                    canMoveBack={currentQuestionIndex > 0}
                    canMoveNext={Boolean(currentAnswer)}
                    isLastQuestion={isLastQuestion}
                    isSubmitting={isSubmitting}
                  />
                </>
              )}
            </div>
          </div>
        </aside>

        <div className="advisor-main">
          <SectionCard
            title="Risk Profile"
            subtitle="The investor profile appears only after the full questionnaire is submitted."
          >
            {!recommendation || !investorProfile ? (
              <ResultPlaceholder
                title="Profile locked until submission"
                body="Complete the questionnaire flow in the sidebar and click Submit. We will then show preference, capacity, target volatility, and calibrated A here."
              />
            ) : (
              <div className="stack-block">
                <div className="metric-grid metric-grid-profile">
                  <MetricCard label="Preference Score" value={formatNumber(investorProfile.risk_preference_score)} />
                  <MetricCard label="Capacity Score" value={formatNumber(investorProfile.risk_capacity_score)} />
                  <MetricCard label="Risk Category" value={investorProfile.risk_category ?? investorProfile.investor_type} />
                  <MetricCard label="Risk Aversion A" value={formatNumber(investorProfile.risk_aversion_A)} />
                  <MetricCard label="Portfolio Volatility" value={formatPercent(investorProfile.calibrated_portfolio_volatility)} />
                </div>
                <div className="construct-grid">
                  {recommendation.constructSummary.map((item) => (
                    <div key={item.construct} className="construct-card">
                      <strong>{formatLabel(item.construct)}</strong>
                      <p>{item.interpretation}</p>
                      <div className="construct-metrics">
                        <span>Score: {formatNumber(item.normalized_score)}</span>
                        <span>Target Vol: {formatPercent(item.target_volatility)}</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="answer-summary answer-summary-compact">
                  <h3>Selected Answers</h3>
                  <div className="answer-chip-grid answer-chip-grid-questions">
                    {selectedRows.map((row) => (
                      <div key={row.question_id} className="answer-chip answer-chip-question" tabIndex={0}>
                        <div className="answer-question-copy">
                          <span>{row.question_id.toUpperCase()}</span>
                          <strong>{row.question}</strong>
                        </div>
                        <div className="answer-chip-reveal">
                          <span>Selected {row.selected_option}</span>
                          <strong>{row.selected_text}</strong>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </SectionCard>

          <SectionCard
            title="Portfolio Output"
            subtitle="The recommendation is generated after submission, not while the questionnaire is still in progress."
            action={<SegmentedTabs options={RIGHT_TABS} activeId={rightTab} onChange={setRightTab} />}
          >
            {!recommendation ? (
              <ResultPlaceholder
                title="Recommendation pending"
                body="No allocation is shown on entry. Submit the completed questionnaire to reveal the recommended weights and portfolio performance."
              />
            ) : rightTab === "allocation" ? (
              <div className="stack-block">
                <div className="allocation-compare-grid">
                  <AllocationPanel
                    title="Recommended Portfolio (No Short Sales)"
                    subtitle="Client-facing default recommendation"
                    portfolio={recommendedPortfolio}
                    weights={weightSeries}
                    chartTitle="No-Short Weights"
                    chartNote="Recommended allocation after the questionnaire calibration"
                  />
                  {boundedShortPortfolio && boundedShortWeightSeries.length > 0 ? (
                    <AllocationPanel
                      title="Alternative Portfolio (Bounded Short Sales)"
                      subtitle="Research extension with limited shorting"
                      portfolio={boundedShortPortfolio}
                      weights={boundedShortWeightSeries}
                      chartTitle="Bounded-Short Weights"
                      chartNote="Alternative allocation under bounded short-sale constraints"
                    />
                  ) : (
                    <div className="allocation-unavailable">
                      <span className="chart-kicker">Alternative Portfolio</span>
                      <strong>Bounded short solution unavailable</strong>
                      <p>
                        The no-short recommendation is still valid. This run did not produce a stable bounded-short
                        solution, so only the primary weights are shown.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="stack-block">
                <PerformanceMetrics performanceSeries={performanceSeries} />
                <PortfolioPerformanceChart data={performanceSeries} />
              </div>
            )}
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

interface AllocationPanelProps {
  title: string;
  subtitle: string;
  portfolio: RecommendationPortfolio | null;
  weights: RecommendationPayload["recommendation"]["weights"];
  chartTitle: string;
  chartNote: string;
}

function AllocationPanel({ title, subtitle, portfolio, weights, chartTitle, chartNote }: AllocationPanelProps) {
  if (!portfolio) {
    return null;
  }

  return (
    <div className="allocation-panel">
      <div className="allocation-panel-copy">
        <span className="chart-kicker">Portfolio Weights</span>
        <strong>{title}</strong>
        <p>{subtitle}</p>
      </div>
      <div className="metric-grid metric-grid-tight allocation-metric-grid">
        <MetricCard label="Expected Return" value={formatPercent(Number(portfolio.expected_return))} />
        <MetricCard label="Volatility" value={formatPercent(Number(portfolio.volatility))} />
        <MetricCard label="Utility" value={formatNumber(Number(portfolio.utility), 3)} />
        <MetricCard label="Short Exposure" value={formatPercent(Number(portfolio.short_exposure ?? 0))} />
      </div>
      <WeightChart weights={weights} title={chartTitle} note={chartNote} height={320} />
    </div>
  );
}

interface QuestionStepperProps {
  question: QuestionSchema;
  questionIndex: number;
  totalQuestions: number;
  selectedOption?: string;
  onSelect: (questionId: string, value: string) => void;
  onPrevious: () => void;
  onNext: () => void;
  onSubmit: () => void;
  canMoveBack: boolean;
  canMoveNext: boolean;
  isLastQuestion: boolean;
  isSubmitting: boolean;
}

function QuestionStepper({
  question,
  questionIndex,
  totalQuestions,
  selectedOption,
  onSelect,
  onPrevious,
  onNext,
  onSubmit,
  canMoveBack,
  canMoveNext,
  isLastQuestion,
  isSubmitting,
}: QuestionStepperProps) {
  return (
    <div className="question-stepper">
      <div className="stepper-progress-row">
        <span className="stepper-progress-label">
          {question.questionId.toUpperCase()} · {formatLabel(question.construct)}
        </span>
        <strong>
          {questionIndex + 1}/{totalQuestions}
        </strong>
      </div>
      <div className="stepper-track">
        <div
          className="stepper-track-fill"
          style={{ width: `${((questionIndex + 1) / totalQuestions) * 100}%` }}
        />
      </div>
      <div className="question-card question-card-compact">
        <div className="question-heading">
          <strong>{question.title}</strong>
        </div>
        <div className="option-grid">
          {question.options.map((option) => {
            const checked = selectedOption === option.code;
            return (
              <label key={option.code} className={checked ? "option-card option-card-selected" : "option-card"}>
                <input
                  type="radio"
                  name={question.questionId}
                  value={option.code}
                  checked={checked}
                  onChange={(event) => onSelect(question.questionId, event.target.value)}
                />
                <span className="option-code">{option.code}</span>
                <span>{option.text}</span>
              </label>
            );
          })}
        </div>
      </div>
      <div className="stepper-actions">
        <button type="button" className="secondary-action-button" onClick={onPrevious} disabled={!canMoveBack}>
          Previous
        </button>
        {isLastQuestion ? (
          <button
            type="button"
            className="primary-action-button"
            onClick={onSubmit}
            disabled={!canMoveNext || isSubmitting}
          >
            {isSubmitting ? "Submitting..." : "Submit"}
          </button>
        ) : (
          <button
            type="button"
            className="primary-action-button"
            onClick={onNext}
            disabled={!canMoveNext}
          >
            Next
          </button>
        )}
      </div>
    </div>
  );
}

function ResultPlaceholder({ title, body }: { title: string; body: string }) {
  return (
    <div className="result-placeholder">
      <div className="result-placeholder-badge">Locked Until Submit</div>
      <h3>{title}</h3>
      <p>{body}</p>
    </div>
  );
}

function PerformanceMetrics({
  performanceSeries,
}: {
  performanceSeries: RecommendationPayload["recommendation"]["performanceSeries"];
}) {
  const lastPoint = performanceSeries.length > 0 ? performanceSeries[performanceSeries.length - 1] : null;

  if (!lastPoint) {
    return null;
  }

  return (
    <div className="metric-grid metric-grid-tight">
      <MetricCard
        label="No-Short Cum. Return"
        value={formatPercent(lastPoint.noShortCumulativeReturn)}
        hint="Historical path"
      />
      <MetricCard
        label="Bounded-Short Cum. Return"
        value={formatPercent(lastPoint.boundedShortCumulativeReturn)}
        hint="Historical path"
      />
      <MetricCard
        label="Equal-Weighted Cum. Return"
        value={formatPercent(lastPoint.equalWeightCumulativeReturn)}
        hint="Historical path"
      />
      <MetricCard
        label="GMVP Cum. Return"
        value={formatPercent(lastPoint.gmvpCumulativeReturn)}
        hint="Historical path"
      />
    </div>
  );
}
