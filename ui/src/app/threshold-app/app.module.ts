import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

import { ThresholdThresholdAppComponent } from './threshold-app.component';
import { ParameterControlsComponent } from './components/parameter-controls/parameter-controls.component';
import { ResultsDisplayComponent } from './components/results-display/results-display.component';
import { ScoreDistributionChartComponent } from './components/score-distribution-chart/score-distribution-chart.component';
import { PrecisionRecallChartComponent } from './components/precision-recall-chart/precision-recall-chart.component';
import { BenchmarkDataEditorComponent } from './components/benchmark-data-editor/benchmark-data-editor.component';

@NgModule({
  declarations: [
    ThresholdAppComponent,
    ParameterControlsComponent,
    ResultsDisplayComponent,
    BenchmarkDataEditorComponent,
  ],
  imports: [
    BrowserModule,
    FormsModule,
    CommonModule,
    ScoreDistributionChartComponent,
    PrecisionRecallChartComponent
  ],
  providers: [],
  bootstrap: [ThresholdAppComponent]
})
export class AppModule { }
