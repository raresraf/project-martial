import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { MatGridListModule } from '@angular/material/grid-list';
import {MatCardModule} from '@angular/material/card';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { DiffComponent } from './diff/diff.component';

import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatToolbarModule } from '@angular/material/toolbar'; 
import { FormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations'; 
import {HttpClientModule} from '@angular/common/http';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatSliderModule} from '@angular/material/slider';
import {MatRadioModule} from '@angular/material/radio'; 


import { ThresholdAppComponent } from './threshold-app/threshold-app.component';
import { ParameterControlsComponent } from './threshold-app/components/parameter-controls/parameter-controls.component';
import { ResultsDisplayComponent } from './threshold-app/components/results-display/results-display.component';
import { BenchmarkDataEditorComponent } from './threshold-app/components/benchmark-data-editor/benchmark-data-editor.component';
import { ScoreDistributionChartComponent } from './threshold-app/components/score-distribution-chart/score-distribution-chart.component';
import { PrecisionRecallChartComponent } from './threshold-app/components/precision-recall-chart/precision-recall-chart.component';


@NgModule({
  declarations: [
    AppComponent,
    DiffComponent,
    ThresholdAppComponent,
    ParameterControlsComponent,
    ResultsDisplayComponent,
    BenchmarkDataEditorComponent,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    MatGridListModule,
    MatCardModule,
    CommonModule,
    MatIconModule,
    MatButtonModule,
    MatInputModule,
    MatFormFieldModule,
    MatProgressBarModule,
    MatToolbarModule,
    FormsModule,
    BrowserAnimationsModule,
    HttpClientModule,
    MatCheckboxModule,
    MatSliderModule,
    MatRadioModule,
    ScoreDistributionChartComponent,
    PrecisionRecallChartComponent,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }


