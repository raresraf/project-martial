import { Component, Input } from '@angular/core';
import { CalculationResults } from '../../logic/model';

@Component({
  selector: 'app-results-display',
  templateUrl: './results-display.component.html',
  styleUrls: ['./results-display.component.css']
})
export class ResultsDisplayComponent {
  @Input() results: CalculationResults | null = null;
  
  Math = Math;

  getPrecision(results: CalculationResults | null): number {
    if (!results) return 0;
    const totalProjected = results.projected_TP + results.projected_FP;
    return totalProjected > 0 ? results.projected_TP / totalProjected : 0;
  }
}