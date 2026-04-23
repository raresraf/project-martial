import { Component } from '@angular/core';
import { CORPUS, SIM_COLUMNS, SIM_ROWS, SVM_MISCLASSIFICATIONS, SimScore, SimRow, CorpusEntry } from './network-traffic-data';

@Component({
  selector: 'app-network-traffic-dataset',
  templateUrl: './network-traffic-dataset.component.html',
  styleUrls: ['./network-traffic-dataset.component.css'],
})
export class NetworkTrafficDatasetComponent {
  readonly corpus: CorpusEntry[] = CORPUS;
  readonly simColumns: string[] = SIM_COLUMNS;
  readonly simRows: SimRow[] = SIM_ROWS;
  readonly misclassifications = SVM_MISCLASSIFICATIONS;

  readonly datasetUrl = 'https://github.com/raresraf/project-martial';

  selectedGram: 2 | 3 | 4 = 4;
  corpusExpanded = false;

  getScore(cell: SimScore | null): string {
    if (cell === null) return '—';
    const v = this.selectedGram === 2 ? cell.g2 : this.selectedGram === 3 ? cell.g3 : cell.g4;
    return v.toFixed(2);
  }

  getCellBg(cell: SimScore | null): string {
    if (cell === null) return '#f5f5f5';
    const s = cell.g4;
    if (s >= 0.67) return '#a5d6a7';
    if (s >= 0.60) return '#c8e6c9';
    if (s >= 0.25) return '#ffcdd2';
    if (s >= 0.05) return '#ef9a9a';
    return '#e57373';
  }

  getCellTextColor(cell: SimScore | null): string {
    if (cell === null) return '#9e9e9e';
    return cell.g4 >= 0.60 ? '#1b5e20' : '#7f0000';
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
