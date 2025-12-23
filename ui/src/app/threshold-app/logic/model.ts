export interface BenchmarkDataPoint {
  theta: number;
  y_TP: number;
  y_FP: number;
}

export const benchmarkData: BenchmarkDataPoint[] = [
  { theta: 0.00, y_TP: 179998, y_FP: 89793 },
  { theta: 0.42, y_TP: 176747, y_FP: 45863 },
  { theta: 0.44, y_TP: 174760, y_FP: 41339 },
  { theta: 0.46, y_TP: 167847, y_FP: 37197 },
  { theta: 0.48, y_TP: 164382, y_FP: 33312 },
  { theta: 0.50, y_TP: 159420, y_FP: 29540 },
  { theta: 0.52, y_TP: 153863, y_FP: 26144 },
  { theta: 0.54, y_TP: 147054, y_FP: 22944 },
  { theta: 0.56, y_TP: 139992, y_FP: 19961 },
  { theta: 0.58, y_TP: 132668, y_FP: 17345 },
  { theta: 1.00, y_TP: 0, y_FP: 0 },
].sort((a, b) => a.theta - b.theta);

export interface ScenarioParameters {
  N: number;
  B: number;
  T_TP: number;
  T_FP: number;
  r: number;
}

export interface CalculationResults {
  phiValues: { theta: number; phi: number }[];
  optimalTheta: BenchmarkDataPoint | null;
  projected_TP: number;
  projected_FP: number;
  cost: number;
  tilde_TP_values: { theta: number; value: number }[];
  tilde_FP_values: { theta: number; value: number }[];
}

export function calculateResults(
  params: ScenarioParameters,
  data: BenchmarkDataPoint[]
): CalculationResults {
  const { N, B, T_TP, T_FP, r } = params;

  const y_POS = data[0].y_TP;
  const y_NEG = data[0].y_FP;
  const y_r = y_POS / y_NEG;
  const y_T = y_POS + y_NEG;

  const totalPairs = (N * (N - 1)) / 2;
  const POS = totalPairs * r / (r + 1);
  const NEG = totalPairs / (r + 1);

  const c_S = ((y_r + 1) * N * (N - 1)) / (2 * (r + 1) * y_T * y_r);

  const tilde_TP_values = data.map(point => {
    const hat_TP = point.y_TP / y_POS;
    return { theta: point.theta, value: hat_TP * POS };
  });

  const tilde_FP_values = data.map(point => {
    const hat_FP = point.y_FP / y_NEG;
    return { theta: point.theta, value: hat_FP * NEG };
  });

  const phiValues = data.map(point => {
    const phi = c_S * (point.y_TP * r * T_TP + point.y_FP * y_r * T_FP);
    return { theta: point.theta, phi };
  });

  let low = 0;
  let high = phiValues.length - 1;
  let optimalTheta: BenchmarkDataPoint | null = null;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (phiValues[mid].phi <= B) {
      optimalTheta = data[mid];
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }

  let projected_TP = 0;
  let projected_FP = 0;
  let cost = 0;

  if (optimalTheta) {
    const hat_TP = optimalTheta.y_TP / y_POS;
    const hat_FP = optimalTheta.y_FP / y_NEG;

    projected_TP = hat_TP * POS;
    projected_FP = hat_FP * NEG;
    cost = projected_TP * T_TP + projected_FP * T_FP;
  }

  return {
    phiValues,
    optimalTheta,
    projected_TP,
    projected_FP,
    cost,
    tilde_TP_values,
    tilde_FP_values,
  };
}
