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

  entries: LeaderboardEntry[] = [
    {
      rank: 1,
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
