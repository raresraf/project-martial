import { Component, OnInit } from '@angular/core';
import {
  ScenarioParameters,
  CalculationResults,
  calculateResults,
  benchmarkData,
  BenchmarkDataPoint,
} from './logic/model';

@Component({
  selector: 'app-threshold-app',
  templateUrl: './threshold-app.component.html',
  styleUrls: ['./threshold-app.component.css']
})
export class ThresholdAppComponent implements OnInit {
  title = 'threshold-app';

  parameters: ScenarioParameters = {
    N: 100,
    B: 24000,
    T_TP: 120,
    T_FP: 15,
    r: 0.01,
  };

  results: CalculationResults | null = null;
  
  benchmarkDataPoints: BenchmarkDataPoint[] = [...benchmarkData];

  benchmarkYPos: number = 0;
  benchmarkYNeg: number = 0;
  benchmarkYR: number = 0;
  benchmarkYT: number = 0;

  ngOnInit(): void {
    this.recalculateBenchmarkStats();
    this.calculate();
  }

  calculate(): void {
    if (this.benchmarkDataPoints.length === 0) {
      this.results = null;
      return;
    }
    this.results = calculateResults(this.parameters, this.benchmarkDataPoints);
  }

  handleParameterChange(event: { paramName: keyof ScenarioParameters, value: number }): void {
    this.parameters = {
      ...this.parameters,
      [event.paramName]: event.value,
    };
    this.calculate();
  }

  handleBenchmarkDataChange(newData: BenchmarkDataPoint[]): void {
    this.benchmarkDataPoints = newData.sort((a, b) => a.theta - b.theta);
    this.recalculateBenchmarkStats();
    this.calculate();
  }
  
  recalculateBenchmarkStats(): void {
    if (this.benchmarkDataPoints.length > 0) {
      const y_POS_init = this.benchmarkDataPoints[0].y_TP;
      const y_NEG_init = this.benchmarkDataPoints[0].y_FP;
      this.benchmarkYPos = y_POS_init;
      this.benchmarkYNeg = y_NEG_init;
      this.benchmarkYR = y_POS_init / y_NEG_init;
      this.benchmarkYT = y_POS_init + y_NEG_init;
    } else {
      this.benchmarkYPos = 0;
      this.benchmarkYNeg = 0;
      this.benchmarkYR = 0;
      this.benchmarkYT = 0;
    }
  }
}
