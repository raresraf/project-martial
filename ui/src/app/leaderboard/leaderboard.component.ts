import { Component } from '@angular/core';

export interface LeaderboardEntry {
  rank: number;
  model: string;
  category: string;
  bestF1: number;
  precision: number;
  recall: number;
  accuracy: number;
  fpr: number;
  optimalThreshold: string;
  link?: string;
}

@Component({
  selector: 'app-leaderboard',
  templateUrl: './leaderboard.component.html',
  styleUrls: ['./leaderboard.component.css']
})
export class LeaderboardComponent {
  readonly datasetSize = 4000;
  readonly similarPairs = 2000;
  readonly notSimilarPairs = 2000;
  readonly datasetUrl = 'https://github.com/raresraf/project-martial/tree/main/dataset/CoSiM';

  readonly entries: LeaderboardEntry[] = [
    {
      rank: 1,
      model: 'Gemini 2.5 Pro (r=1)',
      category: 'LLM + USE',
      bestF1: 0.9395,
      precision: 0.9393,
      recall: 0.9399,
      accuracy: 0.9395,
      fpr: 0.0601,
      optimalThreshold: '0.31',
      link: 'https://github.com/raresraf/project-martial',
    },
    {
      rank: 2,
      model: 'Gemini 2.5 Pro (r=3)',
      category: 'LLM + USE',
      bestF1: 0.9369,
      precision: 0.9368,
      recall: 0.9369,
      accuracy: 0.9370,
      fpr: 0.0631,
      optimalThreshold: '0.74',
      link: 'https://github.com/raresraf/project-martial',
    },
    {
      rank: 3,
      model: 'Gemini 2.5 Pro (r=6)',
      category: 'LLM + USE',
      bestF1: 0.9177,
      precision: 0.9175,
      recall: 0.9182,
      accuracy: 0.9178,
      fpr: 0.0818,
      optimalThreshold: '0.94',
      link: 'https://github.com/raresraf/project-martial',
    },
    {
      rank: 4,
      model: 'Gemini 2.5 Flash (r=6)',
      category: 'LLM + USE',
      bestF1: 0.8974,
      precision: 0.8994,
      recall: 0.9005,
      accuracy: 0.8974,
      fpr: 0.0995,
      optimalThreshold: '0.62',
      link: 'https://github.com/raresraf/project-martial',
    },
    {
      rank: 5,
      model: 'Gemini 2.5 Flash (r=3)',
      category: 'LLM + USE',
      bestF1: 0.8897,
      precision: 0.8928,
      recall: 0.8933,
      accuracy: 0.8897,
      fpr: 0.1067,
      optimalThreshold: '0.56',
      link: 'https://github.com/raresraf/project-martial',
    },
    {
      rank: 6,
      model: 'Gemini 2.5 Flash (r=1)',
      category: 'LLM + USE',
      bestF1: 0.8798,
      precision: 0.8810,
      recall: 0.8823,
      accuracy: 0.8799,
      fpr: 0.1177,
      optimalThreshold: '0.44',
      link: 'https://github.com/raresraf/project-martial',
    },
    {
      rank: 7,
      model: 'Complexity-Based Birthmarks',
      category: 'Dynamic Analysis',
      bestF1: 0.8200,
      precision: 0.8500,
      recall: 0.9000,
      accuracy: 0.8300,
      fpr: 0.3200,
      optimalThreshold: '0.50',
      link: 'https://github.com/raresraf/project-martial',
    },
    {
      rank: 8,
      model: 'MOSS',
      category: 'Syntax-based',
      bestF1: 0.7230,
      precision: 0.9130,
      recall: 0.5985,
      accuracy: 0.7708,
      fpr: 0.0570,
      optimalThreshold: '10%',
      link: 'https://theory.stanford.edu/~aiken/moss/',
    },
  ];

  birthmarksExpanded = false;
  mossExpanded = false;
  proExpanded = false;
  flashExpanded = false;

  scrollToDisclaimer() {
    document.getElementById('disclaimer')?.scrollIntoView({ behavior: 'smooth' });
  }

  scrollToLlmDisclaimer() {
    document.getElementById('llm-disclaimer')?.scrollIntoView({ behavior: 'smooth' });
  }

  readonly birthmarkThresholdData = [
    { threshold: 0.42, precision: 0.800, tpr: 0.980, f1: 0.815 },
    { threshold: 0.44, precision: 0.820, tpr: 0.970, f1: 0.820 },
    { threshold: 0.46, precision: 0.830, tpr: 0.940, f1: 0.822 },
    { threshold: 0.48, precision: 0.840, tpr: 0.920, f1: 0.824 },
    { threshold: 0.50, precision: 0.850, tpr: 0.900, f1: 0.827 },
    { threshold: 0.52, precision: 0.865, tpr: 0.855, f1: 0.822 },
    { threshold: 0.54, precision: 0.880, tpr: 0.830, f1: 0.809 },
    { threshold: 0.56, precision: 0.890, tpr: 0.790, f1: 0.795 },
    { threshold: 0.58, precision: 0.892, tpr: 0.750, f1: 0.784 },
  ];

  readonly geminiProR1ThresholdData = [
    { threshold: 0.20, precision: 0.9060, recall: 0.8931, f1: 0.8950, accuracy: 0.8964 },
    { threshold: 0.25, precision: 0.9311, recall: 0.9280, f1: 0.9290, accuracy: 0.9293 },
    { threshold: 0.28, precision: 0.9343, recall: 0.9337, f1: 0.9339, accuracy: 0.9341 },
    { threshold: 0.30, precision: 0.9390, recall: 0.9393, f1: 0.9391, accuracy: 0.9392 },
    { threshold: 0.31, precision: 0.9393, recall: 0.9399, f1: 0.9395, accuracy: 0.9395 },
    { threshold: 0.33, precision: 0.9388, recall: 0.9397, f1: 0.9389, accuracy: 0.9389 },
    { threshold: 0.35, precision: 0.9379, recall: 0.9386, f1: 0.9373, accuracy: 0.9373 },
    { threshold: 0.40, precision: 0.9241, recall: 0.9215, f1: 0.9187, accuracy: 0.9187 },
    { threshold: 0.45, precision: 0.9062, recall: 0.8973, f1: 0.8932, accuracy: 0.8932 },
    { threshold: 0.50, precision: 0.8781, recall: 0.8566, f1: 0.8506, accuracy: 0.8506 },
  ];

  readonly geminiFlashR6ThresholdData = [
    { threshold: 0.55, precision: 0.8292, recall: 0.8200, f1: 0.8218, accuracy: 0.8245 },
    { threshold: 0.58, precision: 0.8738, recall: 0.8740, f1: 0.8739, accuracy: 0.8744 },
    { threshold: 0.60, precision: 0.8935, recall: 0.8952, f1: 0.8935, accuracy: 0.8936 },
    { threshold: 0.61, precision: 0.8967, recall: 0.8982, f1: 0.8957, accuracy: 0.8958 },
    { threshold: 0.62, precision: 0.8994, recall: 0.9005, f1: 0.8974, accuracy: 0.8974 },
    { threshold: 0.63, precision: 0.8955, recall: 0.8953, f1: 0.8914, accuracy: 0.8914 },
    { threshold: 0.65, precision: 0.8889, recall: 0.8861, f1: 0.8809, accuracy: 0.8810 },
    { threshold: 0.68, precision: 0.8747, recall: 0.8602, f1: 0.8511, accuracy: 0.8519 },
    { threshold: 0.70, precision: 0.8594, recall: 0.8328, f1: 0.8202, accuracy: 0.8223 },
    { threshold: 0.75, precision: 0.7997, recall: 0.7095, f1: 0.6712, accuracy: 0.6906 },
  ];

  readonly mossThresholdData = [
    { threshold: 0,  tpr: 0.6110, fpr: 0.1405, accuracy: 0.7352, precision: 0.8130, f1: 0.6977 },
    { threshold: 5,  tpr: 0.6070, fpr: 0.0855, accuracy: 0.7608, precision: 0.8765, f1: 0.7173 },
    { threshold: 10, tpr: 0.5985, fpr: 0.0570, accuracy: 0.7708, precision: 0.9130, f1: 0.7230 },
    { threshold: 15, tpr: 0.5720, fpr: 0.0340, accuracy: 0.7690, precision: 0.9439, f1: 0.7123 },
    { threshold: 20, tpr: 0.5365, fpr: 0.0170, accuracy: 0.7598, precision: 0.9693, f1: 0.6907 },
    { threshold: 25, tpr: 0.5085, fpr: 0.0085, accuracy: 0.7500, precision: 0.9836, f1: 0.6704 },
    { threshold: 30, tpr: 0.4880, fpr: 0.0040, accuracy: 0.7420, precision: 0.9919, f1: 0.6542 },
    { threshold: 35, tpr: 0.4675, fpr: 0.0015, accuracy: 0.7330, precision: 0.9968, f1: 0.6365 },
    { threshold: 40, tpr: 0.4355, fpr: 0.0010, accuracy: 0.7173, precision: 0.9977, f1: 0.6063 },
    { threshold: 45, tpr: 0.4135, fpr: 0.0010, accuracy: 0.7063, precision: 0.9976, f1: 0.5847 },
    { threshold: 50, tpr: 0.3835, fpr: 0.0000, accuracy: 0.6917, precision: 1.0000, f1: 0.5544 },
    { threshold: 60, tpr: 0.3375, fpr: 0.0000, accuracy: 0.6687, precision: 1.0000, f1: 0.5047 },
    { threshold: 80, tpr: 0.2840, fpr: 0.0000, accuracy: 0.6420, precision: 1.0000, f1: 0.4424 },
    { threshold: 95, tpr: 0.2020, fpr: 0.0000, accuracy: 0.6010, precision: 1.0000, f1: 0.3361 },
  ];
}
