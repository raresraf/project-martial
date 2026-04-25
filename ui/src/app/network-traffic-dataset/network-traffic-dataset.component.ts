import { Component } from '@angular/core';
import { CORPUS, SIM_COLUMNS, SIM_ROWS, FULL_MESH_COLS, FULL_MESH_ROWS, NOALPHA_MESH_ROWS, SVM_MISCLASSIFICATIONS, SimScore, SimRow, CorpusEntry } from './network-traffic-data';

@Component({
  selector: 'app-network-traffic-dataset',
  templateUrl: './network-traffic-dataset.component.html',
  styleUrls: ['./network-traffic-dataset.component.css'],
})
export class NetworkTrafficDatasetComponent {
  readonly corpus: CorpusEntry[] = CORPUS;
  readonly simColumns: string[] = SIM_COLUMNS;
  readonly simRows: SimRow[] = SIM_ROWS;
  readonly fullMeshCols: string[] = FULL_MESH_COLS;
  readonly fullMeshRows: SimRow[] = FULL_MESH_ROWS;
  readonly noAlphaMeshRows: SimRow[] = NOALPHA_MESH_ROWS;
  readonly misclassifications = SVM_MISCLASSIFICATIONS;

  readonly datasetUrl = 'https://github.com/raresraf/project-martial';

  selectedGram: 2 | 3 | 4 = 4;
  corpusExpanded = false;
  fullMeshExpanded = false;
  activeDataset: 'all' | 'noalpha' = 'all';

  get activeMeshRows(): SimRow[] {
    return this.activeDataset === 'all' ? this.fullMeshRows : this.noAlphaMeshRows;
  }

  get activePartialRows(): SimRow[] {
    if (this.activeDataset === 'all') return this.simRows;
    return this.noAlphaMeshRows.map(r => ({ ...r, scores: r.scores.slice(0, 6) }));
  }

  getScore(cell: SimScore | null): string {
    if (cell === null) return '—';
    const v = this.selectedGram === 2 ? cell.g2 : this.selectedGram === 3 ? cell.g3 : cell.g4;
    return v.toFixed(2);
  }

  getCellBg(cell: SimScore | null): string {
    if (cell === null) return '#f5f5f5';
    const s = cell.g4;
    if (this.activeDataset === 'noalpha') {
      if (s >= 0.55) return '#a5d6a7';
      if (s >= 0.40) return '#c8e6c9';
      if (s >= 0.20) return '#ffcdd2';
      if (s >= 0.05) return '#ef9a9a';
      return '#e57373';
    }
    if (s >= 0.67) return '#a5d6a7';
    if (s >= 0.60) return '#c8e6c9';
    if (s >= 0.35) return '#dcedc8';
    if (s >= 0.25) return '#ffcdd2';
    if (s >= 0.05) return '#ef9a9a';
    return '#e57373';
  }

  getCellTextColor(cell: SimScore | null): string {
    if (cell === null) return '#9e9e9e';
    const threshold = this.activeDataset === 'noalpha' ? 0.40 : 0.35;
    return cell.g4 >= threshold ? '#1b5e20' : '#7f0000';
  }

  get legendItems(): Array<{ bg: string; label: string }> {
    if (this.activeDataset === 'noalpha') {
      return [
        { bg: '#a5d6a7', label: '≥ 0.55 — strong similarity' },
        { bg: '#c8e6c9', label: '≥ 0.40 — high similarity' },
        { bg: '#ffcdd2', label: '≥ 0.20 — low similarity' },
        { bg: '#ef9a9a', label: '≥ 0.05 — very low' },
        { bg: '#e57373', label: '< 0.05 — extremely low' },
      ];
    }
    return [
      { bg: '#a5d6a7', label: '≥ 0.67 — strong similarity' },
      { bg: '#c8e6c9', label: '≥ 0.60 — high similarity' },
      { bg: '#dcedc8', label: '≥ 0.35 — moderate similarity' },
      { bg: '#ffcdd2', label: '≥ 0.25 — low similarity' },
      { bg: '#ef9a9a', label: '≥ 0.05 — very low' },
      { bg: '#e57373', label: '< 0.05 — extremely low' },
    ];
  }

  private static readonly GROUP_LABELS: Record<'mysql' | 'postgresql' | 'sqlserver', string> = {
    mysql: 'MySQL',
    postgresql: 'PostgreSQL',
    sqlserver: 'SQL Server',
  };

  getGroupLabel(group: 'mysql' | 'postgresql' | 'sqlserver'): string {
    return NetworkTrafficDatasetComponent.GROUP_LABELS[group];
  }

  getEngineLabel(row: SimRow): string {
    return row.engine.slice(NetworkTrafficDatasetComponent.GROUP_LABELS[row.group].length + 1);
  }

  isFirstInGroup<T extends { group: string }>(items: T[], i: number): boolean {
    return i > 0 && items[i - 1].group !== items[i].group;
  }
}
