export interface LangStats {
  language: string;
  pairs: number;
  totalLoc: number;
  fileCount: number;
  avgLoc: number;
}

export interface MossResult {
  threshold: number;
  tpr: number;
  fpr: number;
  accuracy: number;
  precision: number;
  f1: number;
}

export interface IndexRange {
  from: number;
  to: number;
  language: string;
  source: string;
  pillar: 'open-source' | 'student' | 'competitive';
}

export interface Pair {
  index: number;
  label: 'similar' | 'notsimilar';
  language: string;
  source: string;
  pillar: 'open-source' | 'student' | 'competitive';
}

export const INDEX_RANGES: IndexRange[] = [
  { from: 0,    to: 99,   language: 'Go',         source: 'Kubernetes',            pillar: 'open-source' },
  { from: 100,  to: 199,  language: 'TypeScript',  source: 'VSCode',                pillar: 'open-source' },
  { from: 200,  to: 299,  language: 'C',           source: 'Linux kernel',          pillar: 'open-source' },
  { from: 300,  to: 399,  language: 'Java',        source: 'Elastic',               pillar: 'open-source' },
  { from: 400,  to: 499,  language: 'Rust',        source: 'Servo',                 pillar: 'open-source' },
  { from: 500,  to: 599,  language: 'C',           source: 'Student assignment #1', pillar: 'student' },
  { from: 600,  to: 699,  language: 'CUDA',        source: 'Student assignment #2', pillar: 'student' },
  { from: 700,  to: 799,  language: 'Python',      source: 'Student assignment #3', pillar: 'student' },
  { from: 800,  to: 899,  language: 'Python',      source: 'Student assignment #4', pillar: 'student' },
  { from: 900,  to: 999,  language: 'Python',      source: 'Student assignment #5', pillar: 'student' },
  { from: 1000, to: 1099, language: 'Python',      source: 'Student assignment #6', pillar: 'student' },
  { from: 1100, to: 1199, language: 'OpenCL',      source: 'Student assignment #7', pillar: 'student' },
  { from: 1200, to: 1999, language: 'C++',         source: 'Codeforces',            pillar: 'competitive' },
];

// Per-language stats — from paper Table 3 (total across similar + not-similar)
export const LANG_STATS: LangStats[] = [
  { language: 'C',          pairs: 200, totalLoc: 4655446, fileCount: 3626, avgLoc: 1283.9 },
  { language: 'Rust',       pairs: 100, totalLoc: 1301474, fileCount: 1304, avgLoc: 998.1  },
  { language: 'Go',         pairs: 100, totalLoc:  708353, fileCount: 1398, avgLoc:  506.7  },
  { language: 'TypeScript', pairs: 100, totalLoc:  549249, fileCount:  947, avgLoc:  579.9  },
  { language: 'Java',       pairs: 100, totalLoc:  518231, fileCount: 1050, avgLoc:  493.6  },
  { language: 'Python',     pairs: 400, totalLoc:  173655, fileCount:  826, avgLoc:  210.2  },
  { language: 'C++',        pairs: 800, totalLoc:  134264, fileCount: 2185, avgLoc:   61.5  },
  { language: 'OpenCL',     pairs: 100, totalLoc:   98697, fileCount:  120, avgLoc:  822.5  },
  { language: 'CUDA',       pairs: 100, totalLoc:   74360, fileCount:  195, avgLoc:  381.3  },
];

// MOSS baseline results — from paper Table 2 + Figure 6 data points
export const MOSS_RESULTS: MossResult[] = [
  { threshold:  0, tpr: 0.6110, fpr: 0.1405, accuracy: 0.7352, precision: 0.8130, f1: 0.6977 },
  { threshold:  5, tpr: 0.6070, fpr: 0.0855, accuracy: 0.7608, precision: 0.8765, f1: 0.7173 },
  { threshold: 10, tpr: 0.5985, fpr: 0.0570, accuracy: 0.7708, precision: 0.9130, f1: 0.7230 },
  { threshold: 15, tpr: 0.5720, fpr: 0.0340, accuracy: 0.7690, precision: 0.9439, f1: 0.7123 },
  { threshold: 20, tpr: 0.5365, fpr: 0.0170, accuracy: 0.7598, precision: 0.9693, f1: 0.6907 },
  { threshold: 25, tpr: 0.5085, fpr: 0.0085, accuracy: 0.7500, precision: 0.9836, f1: 0.6704 },
  { threshold: 30, tpr: 0.4880, fpr: 0.0040, accuracy: 0.7420, precision: 0.9919, f1: 0.6542 },
  { threshold: 35, tpr: 0.4675, fpr: 0.0015, accuracy: 0.7330, precision: 0.9968, f1: 0.6365 },
  { threshold: 40, tpr: 0.4355, fpr: 0.0010, accuracy: 0.7173, precision: 0.9977, f1: 0.6063 },
  { threshold: 45, tpr: 0.4135, fpr: 0.0010, accuracy: 0.7063, precision: 0.9976, f1: 0.5847 },
  { threshold: 50, tpr: 0.3835, fpr: 0.0000, accuracy: 0.6917, precision: 1.0000, f1: 0.5544 },
  { threshold: 55, tpr: 0.3610, fpr: 0.0000, accuracy: 0.6805, precision: 1.0000, f1: 0.5305 },
  { threshold: 60, tpr: 0.3375, fpr: 0.0000, accuracy: 0.6687, precision: 1.0000, f1: 0.5047 },
  { threshold: 65, tpr: 0.3200, fpr: 0.0000, accuracy: 0.6600, precision: 1.0000, f1: 0.4848 },
  { threshold: 70, tpr: 0.3085, fpr: 0.0000, accuracy: 0.6542, precision: 1.0000, f1: 0.4715 },
  { threshold: 75, tpr: 0.2960, fpr: 0.0000, accuracy: 0.6480, precision: 1.0000, f1: 0.4568 },
  { threshold: 80, tpr: 0.2840, fpr: 0.0000, accuracy: 0.6420, precision: 1.0000, f1: 0.4424 },
  { threshold: 85, tpr: 0.2715, fpr: 0.0000, accuracy: 0.6357, precision: 1.0000, f1: 0.4271 },
  { threshold: 90, tpr: 0.2440, fpr: 0.0000, accuracy: 0.6220, precision: 1.0000, f1: 0.3923 },
  { threshold: 95, tpr: 0.2020, fpr: 0.0000, accuracy: 0.6010, precision: 1.0000, f1: 0.3361 },
];
