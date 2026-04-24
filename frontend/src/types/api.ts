export interface FundReturnPoint {
  date: string;
  return: number;
}

export interface FundRecord {
  fund_name: string;
  sleeve: string;
  fund_house: string;
  share_class_currency: string;
  annual_fee_pct: number;
  selection_reason: string;
  short_label: string;
  return_series: FundReturnPoint[];
}

export interface FundSelectorOption {
  fundName: string;
  shortLabel: string;
  sleeve: string;
}

export interface FrontierPoint {
  expected_return: number;
  volatility: number;
}

export interface AssetPoint {
  fund_name: string;
  short_label: string;
  expected_return: number;
  volatility: number;
}

export interface PortfolioSnapshot {
  portfolio_name: string;
  expected_return: number;
  volatility: number;
  utility: number;
  max_drawdown: number;
  gross_exposure?: number;
  client_ready?: boolean;
  top_long_positions?: string;
  top_short_positions?: string;
  weighted_long_fee_pct?: number;
}

export interface PortfolioPerformancePoint {
  date: string;
  equalWeightReturn: number;
  equalWeightCumulativeReturn: number;
  gmvpReturn: number;
  gmvpCumulativeReturn: number;
  noShortReturn: number;
  noShortCumulativeReturn: number;
  boundedShortReturn?: number | null;
  boundedShortCumulativeReturn?: number | null;
}

export interface FrontierPanel {
  title: string;
  frontier: FrontierPoint[];
  gmvp: PortfolioSnapshot;
  gmvpWeights: WeightPoint[];
  optimal: PortfolioSnapshot;
}

export interface Part1Payload {
  summary: {
    recommendedPolicy: string;
    dateStart: string;
    dateEnd: string;
    observationCount: number;
    investorType: string;
    riskAversionA: number;
  };
  availableFunds: FundSelectorOption[];
  selectedFundNames: string[];
  funds: FundRecord[];
  assetPoints: AssetPoint[];
  noShortFrontier: FrontierPanel;
  shortSalesFrontier: FrontierPanel;
  benchmark: PortfolioSnapshot[];
}

export interface QuestionOption {
  code: string;
  text: string;
  score: number;
}

export interface QuestionSchema {
  questionId: string;
  construct: string;
  weight: number;
  title: string;
  options: QuestionOption[];
}

export interface QuestionRow {
  question_id: string;
  question: string;
  construct: string;
  selected_option: string;
  selected_text: string;
  raw_score: number;
  question_weight: number;
  normalized_construct_weight: number;
  construct_contribution: number;
}

export interface ConstructSummary {
  construct: string;
  question_count: number;
  raw_weight_sum: number;
  normalized_score: number;
  target_volatility: number;
  volatility_band: string;
  interpretation: string;
}

export interface InvestorProfile {
  total_raw_score: number;
  risk_preference_score: number;
  risk_capacity_score: number;
  target_vol_pref: number;
  target_vol_cap: number;
  final_target_vol: number;
  final_risk_score?: number;
  risk_category?: string;
  binding_construct: string;
  risk_aversion_A: number;
  risk_aversion_lower_A?: number;
  risk_aversion_upper_A?: number;
  calibrated_portfolio_volatility: number;
  calibration_gap: number;
  calibration_method?: string;
  investor_type: string;
}

export interface WeightPoint {
  fundName: string;
  shortLabel: string;
  sleeve: string;
  weight: number;
}

export interface RecommendationPayload {
  questionRows: QuestionRow[];
  constructSummary: ConstructSummary[];
  investorProfile: InvestorProfile;
  recommendation: {
    portfolio: Record<string, string | number | boolean | null>;
    weights: WeightPoint[];
    boundedShortPortfolio?: Record<string, string | number | boolean | null> | null;
    boundedShortWeights?: WeightPoint[] | null;
    comparison: PortfolioSnapshot[];
    performanceSeries: PortfolioPerformancePoint[];
    topLongPositions: string;
    topShortPositions: string;
  };
}

export interface AdvisorBootstrapPayload {
  questionnaire: QuestionSchema[];
}
