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
      rank: 2,
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

  scrollToDisclaimer() {
    document.getElementById('disclaimer')?.scrollIntoView({ behavior: 'smooth' });
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
