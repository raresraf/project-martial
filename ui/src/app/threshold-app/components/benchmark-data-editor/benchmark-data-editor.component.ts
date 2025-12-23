import { Component, Input, Output, EventEmitter } from '@angular/core';
import { BenchmarkDataPoint, benchmarkData as defaultBenchmarkData } from '../../logic/model';

@Component({
  selector: 'app-benchmark-data-editor',
  templateUrl: './benchmark-data-editor.component.html',
  styleUrls: ['./benchmark-data-editor.component.css']
})
export class BenchmarkDataEditorComponent {
  @Input() benchmarkData: BenchmarkDataPoint[] = [];
  @Output() benchmarkDataChange = new EventEmitter<BenchmarkDataPoint[]>();

  updateData(): void {
    this.benchmarkDataChange.emit(this.benchmarkData);
  }

  addRow(): void {
    this.benchmarkData.push({ theta: 0, y_TP: 0, y_FP: 0 });
    this.updateData();
  }

  deleteRow(index: number): void {
    this.benchmarkData.splice(index, 1);
    this.updateData();
  }

  restoreDefaults(): void {
    this.benchmarkData = JSON.parse(JSON.stringify(defaultBenchmarkData));
    this.updateData();
  }
}