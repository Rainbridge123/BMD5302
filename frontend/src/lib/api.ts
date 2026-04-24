import type {
  AdvisorBootstrapPayload,
  Part1Payload,
  RecommendationPayload,
} from "../types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const errorBody = (await response.json().catch(() => ({}))) as { error?: string };
    throw new Error(errorBody.error ?? `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function getPart1Data() {
  return requestJson<Part1Payload>("/api/part1");
}

export function getPart1DataForFunds(selectedFunds: string[]) {
  const params = new URLSearchParams();
  for (const fundName of selectedFunds) {
    params.append("fund", fundName);
  }
  return requestJson<Part1Payload>(`/api/part1?${params.toString()}`);
}

export function getAdvisorBootstrap() {
  return requestJson<AdvisorBootstrapPayload>("/api/advisor/bootstrap");
}

export function getRecommendation(answers: Record<string, string>) {
  return requestJson<RecommendationPayload>("/api/advisor/recommendation", {
    method: "POST",
    body: JSON.stringify({ answers }),
  });
}
