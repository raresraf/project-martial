import { Component, AfterViewInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import {
  Chart,
  BarController, BarElement,
  CategoryScale, LinearScale, LogarithmicScale,
  LineController, LineElement, PointElement,
  Title, Tooltip, Legend,
} from 'chart.js';
import AnnotationPlugin from 'chartjs-plugin-annotation';
import { forkJoin } from 'rxjs';

import { INDEX_RANGES, LANG_STATS, MOSS_RESULTS, Pair, LangStats, MossResult, IndexRange } from './cosim-dataset-data';
import { CosimService, SnippetMeta } from './cosim.service';

Chart.register(
  BarController, BarElement,
  CategoryScale, LinearScale, LogarithmicScale,
  LineController, LineElement, PointElement,
  Title, Tooltip, Legend,
  AnnotationPlugin,
);

type LabelFilter  = 'all' | 'similar' | 'notsimilar';
type PillarFilter = 'all' | 'open-source' | 'student' | 'competitive';

const LANG_COLORS: Record<string, string> = {
  'Go':         '#00ADD8',
  'TypeScript': '#3178C6',
  'C':          '#6B6B6B',
  'Java':       '#E76F00',
  'Rust':       '#CE412B',
  'CUDA':       '#76B900',
  'Python':     '#3776AB',
  'OpenCL':     '#9C27B0',
  'C++':        '#004482',
};

export interface SnippetView {
  uuid: string;
  source: string;
  files: string[];
  selectedFile: string;
  content: string | null;
  loading: boolean;
  error: string | null;
}

export interface PairView {
  pair: Pair;
  snippetA: SnippetView;
  snippetB: SnippetView;
  loading: boolean;
}

@Component({
  selector: 'app-cosim-dataset',
  templateUrl: './cosim-dataset.component.html',
  styleUrls: ['./cosim-dataset.component.css'],
})
export class CosimDatasetComponent implements AfterViewInit, OnDestroy {
  @ViewChild('pairsChartCanvas') pairsChartRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('locChartCanvas')   locChartRef!:   ElementRef<HTMLCanvasElement>;
  @ViewChild('mossChartCanvas')  mossChartRef!:  ElementRef<HTMLCanvasElement>;

  private charts: Chart[] = [];

  readonly datasetUrl   = 'https://github.com/raresraf/project-martial/tree/main/dataset/CoSiM';
  readonly langStats    : LangStats[]  = LANG_STATS;
  readonly mossResults  : MossResult[] = MOSS_RESULTS;
  readonly indexRanges  : IndexRange[] = INDEX_RANGES;
  readonly languages    : string[]     = ['Go', 'TypeScript', 'C', 'Java', 'Rust', 'CUDA', 'Python', 'OpenCL', 'C++'];

  // Pair browser state
  labelFilter   : LabelFilter  = 'all';
  languageFilter: string       = 'all';
  pillarFilter  : PillarFilter = 'all';
  currentPage   = 0;
  readonly pageSize = 50;

  filteredPairs : Pair[] = [];
  displayedPairs: Pair[] = [];
  totalPages    = 0;

  provenanceExpanded = false;
  statsTableExpanded = false;
  mossTableExpanded  = false;

  // Pair viewer
  activePairView: PairView | null = null;

  private readonly allPairs: Pair[] = this.buildPairs();

  constructor(private cosimSvc: CosimService) { this.applyFilters(); }

  private buildPairs(): Pair[] {
    const pairs: Pair[] = [];
    for (const label of ['similar', 'notsimilar'] as const) {
      for (let i = 0; i < 2000; i++) {
        const range = INDEX_RANGES.find(r => r.from <= i && i <= r.to)!;
        pairs.push({ index: i, label, language: range.language, source: range.source, pillar: range.pillar });
      }
    }
    return pairs;
  }

  // ── filters / pagination ──────────────────────────────────────────────────

  applyFilters(): void {
    this.filteredPairs = this.allPairs.filter(p =>
      (this.labelFilter    === 'all' || p.label    === this.labelFilter)   &&
      (this.languageFilter === 'all' || p.language === this.languageFilter) &&
      (this.pillarFilter   === 'all' || p.pillar   === this.pillarFilter)
    );
    this.totalPages = Math.ceil(this.filteredPairs.length / this.pageSize);
    this.currentPage = 0;
    this.updateDisplayed();
  }

  updateDisplayed(): void {
    const start = this.currentPage * this.pageSize;
    this.displayedPairs = this.filteredPairs.slice(start, start + this.pageSize);
  }

  goToPage(page: number): void {
    if (page < 0 || page >= this.totalPages) return;
    this.currentPage = page;
    this.updateDisplayed();
  }

  get pageNumbers(): number[] {
    const lo = Math.max(0, this.currentPage - 2);
    const hi = Math.min(this.totalPages - 1, this.currentPage + 2);
    const r: number[] = [];
    for (let i = lo; i <= hi; i++) r.push(i);
    return r;
  }

  get allPairsCount(): number { return this.allPairs.length; }

  // ── pair viewer ───────────────────────────────────────────────────────────

  openPair(pair: Pair): void {
    this.activePairView = {
      pair,
      snippetA: { uuid: '', source: '', files: [], selectedFile: '', content: null, loading: true, error: null },
      snippetB: { uuid: '', source: '', files: [], selectedFile: '', content: null, loading: true, error: null },
      loading: true,
    };
    window.scrollTo({ top: 0, behavior: 'smooth' });

    this.cosimSvc.getPairUUIDs(pair.label, pair.index).subscribe({
      next: ({ uuid1, uuid2 }) => {
        this.activePairView!.snippetA.uuid = uuid1;
        this.activePairView!.snippetB.uuid = uuid2;
        this.activePairView!.loading = false;

        forkJoin([
          this.cosimSvc.getSnippetMeta(uuid1),
          this.cosimSvc.getSnippetMeta(uuid2),
        ]).subscribe(([metaA, metaB]) => {
          this.initSnippetView(this.activePairView!.snippetA, metaA);
          this.initSnippetView(this.activePairView!.snippetB, metaB);
        });
      },
      error: err => {
        if (this.activePairView) {
          this.activePairView.loading = false;
          this.activePairView.snippetA.error = 'Failed to load manifest';
          this.activePairView.snippetB.error = 'Failed to load manifest';
          this.activePairView.snippetA.loading = false;
          this.activePairView.snippetB.loading = false;
        }
      },
    });
  }

  private initSnippetView(sv: SnippetView, meta: SnippetMeta | null): void {
    if (!meta) {
      sv.loading = false;
      sv.error = 'Snippet metadata not found in manifest';
      return;
    }
    sv.source = meta.source;
    sv.files  = meta.files;
    if (meta.files.length > 0) {
      sv.selectedFile = meta.files[0];
      this.loadSnippetFile(sv);
    } else {
      sv.loading = false;
      sv.error = 'No code files found for this snippet';
    }
  }

  loadSnippetFile(sv: SnippetView): void {
    if (!sv.selectedFile) return;
    sv.loading = true;
    sv.content = null;
    sv.error   = null;
    this.cosimSvc.loadFile(sv.uuid, sv.selectedFile).subscribe({
      next: text => { sv.content = text; sv.loading = false; },
      error: ()  => { sv.error = `Could not load ${sv.selectedFile}`; sv.loading = false; },
    });
  }

  closePair(): void {
    this.activePairView = null;
  }

  // ── helpers ───────────────────────────────────────────────────────────────

  pillarLabel(pillar: 'open-source' | 'student' | 'competitive'): string {
    const labels: Record<string, string> = {
      'open-source': 'Open Source', 'student': 'Student', 'competitive': 'Competitive',
    };
    return labels[pillar];
  }

  langClass(language: string): string {
    const map: Record<string, string> = {
      'C++': 'cpp', 'C': 'c', 'Go': 'go', 'TypeScript': 'typescript',
      'Java': 'java', 'Rust': 'rust', 'Python': 'python',
      'CUDA': 'cuda', 'OpenCL': 'opencl',
    };
    return 'badge-lang-' + (map[language] ?? language.toLowerCase());
  }

  shortUuid(uuid: string): string { return uuid.slice(0, 8) + '…'; }

  lineCount(content: string | null): number {
    return content ? content.split('\n').length : 0;
  }

  // ── lifecycle ─────────────────────────────────────────────────────────────

  ngAfterViewInit(): void {
    this.initPairsChart();
    this.initLocChart();
    this.initMossChart();
  }

  ngOnDestroy(): void {
    this.charts.forEach(c => c.destroy());
  }

  // ── charts ────────────────────────────────────────────────────────────────

  private initPairsChart(): void {
    const langs = ['Go', 'TypeScript', 'C', 'Java', 'Rust', 'CUDA', 'Python', 'OpenCL', 'C++'];
    const pairCounts: Record<string, number> = {
      Go: 100, TypeScript: 100, C: 200, Java: 100,
      Rust: 100, CUDA: 100, Python: 400, OpenCL: 100, 'C++': 800,
    };

    this.charts.push(new Chart(this.pairsChartRef.nativeElement, {
      type: 'bar',
      data: {
        labels: langs,
        datasets: [{
          label: 'Similar pairs',
          data: langs.map(l => pairCounts[l]),
          backgroundColor: langs.map(l => (LANG_COLORS[l] ?? '#4285F4') + 'BB'),
          borderColor:     langs.map(l =>  LANG_COLORS[l] ?? '#4285F4'),
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Similar pairs per programming language',
            font: { size: 13, family: "'Roboto', sans-serif", weight: 'normal' } },
          tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y} pairs` } },
        },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, title: { display: true, text: 'Number of pairs' } },
        },
      },
    }));
  }

  private initLocChart(): void {
    const sorted = [...LANG_STATS].sort((a, b) => b.totalLoc - a.totalLoc);
    const labels = sorted.map(s => s.language);
    const data   = sorted.map(s => s.totalLoc);

    this.charts.push(new Chart(this.locChartRef.nativeElement, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Total LOC',
          data,
          backgroundColor: labels.map(l => (LANG_COLORS[l] ?? '#4285F4') + 'BB'),
          borderColor:     labels.map(l =>  LANG_COLORS[l] ?? '#4285F4'),
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Total lines of code per language (log scale)',
            font: { size: 13, family: "'Roboto', sans-serif", weight: 'normal' } },
          tooltip: { callbacks: { label: ctx => ` ${(ctx.parsed.y as number).toLocaleString()} lines` } },
        },
        scales: {
          x: { grid: { display: false } },
          y: { type: 'logarithmic', title: { display: true, text: 'Total LOC' } },
        },
      },
    }));
  }

  private initMossChart(): void {
    const labels    = MOSS_RESULTS.map(r => r.threshold + '%');
    const precision = MOSS_RESULTS.map(r => r.precision);
    const recall    = MOSS_RESULTS.map(r => r.tpr);
    const f1        = MOSS_RESULTS.map(r => r.f1);

    this.charts.push(new Chart(this.mossChartRef.nativeElement, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Precision',
            data: precision,
            borderColor: '#34A853',
            backgroundColor: 'rgba(52,168,83,0.08)',
            pointStyle: 'rectRot',
            pointRadius: 4,
            tension: 0.2,
            fill: false,
          },
          {
            label: 'Recall (TPR)',
            data: recall,
            borderColor: '#4285F4',
            backgroundColor: 'rgba(66,133,244,0.08)',
            pointStyle: 'triangle',
            pointRadius: 4,
            tension: 0.2,
            fill: false,
          },
          {
            label: 'F1 Score',
            data: f1,
            borderColor: '#F4B400',
            backgroundColor: 'rgba(244,180,0,0.08)',
            pointStyle: 'circle',
            pointRadius: 5,
            borderWidth: 2.5,
            tension: 0.2,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'top' },
          title: { display: true, text: 'MOSS performance vs detection threshold',
            font: { size: 13, family: "'Roboto', sans-serif", weight: 'normal' } },
          annotation: {
            annotations: {
              bestF1: {
                type: 'line',
                xMin: '10%', xMax: '10%',
                borderColor: 'rgba(244,180,0,0.75)',
                borderWidth: 2,
                borderDash: [6, 4],
                label: {
                  content: 'Best F1 (10%)',
                  display: true,
                  position: 'start',
                  backgroundColor: 'rgba(244,180,0,0.9)',
                  color: '#333',
                  font: { size: 11 },
                  yAdjust: 20,
                },
              },
              precisionMax: {
                type: 'line',
                xMin: '50%', xMax: '50%',
                borderColor: 'rgba(52,168,83,0.75)',
                borderWidth: 2,
                borderDash: [6, 4],
                label: {
                  content: 'Precision = 1.0 (≥50%)',
                  display: true,
                  position: 'end',
                  backgroundColor: 'rgba(52,168,83,0.9)',
                  color: '#fff',
                  font: { size: 11 },
                  yAdjust: 20,
                },
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'Similarity Threshold' },
            grid: { color: 'rgba(0,0,0,0.05)' },
          },
          y: {
            min: 0, max: 1.1,
            title: { display: true, text: 'Score' },
            grid: { color: 'rgba(0,0,0,0.05)' },
          },
        },
      },
    }));
  }
}
