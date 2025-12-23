import { Component, Input, Output, EventEmitter } from '@angular/core';
import { ScenarioParameters } from '../../logic/model';

@Component({
  selector: 'app-parameter-controls',
  templateUrl: './parameter-controls.component.html',
  styleUrls: ['./parameter-controls.component.css']
})
export class ParameterControlsComponent {
  @Input() parameters!: ScenarioParameters;
  @Input() benchmarkYPos: number = 0;
  @Input() benchmarkYNeg: number = 0;
  @Input() benchmarkYR: number = 0;
  @Input() benchmarkYT: number = 0;
  @Output() parameterChange = new EventEmitter<{ paramName: keyof ScenarioParameters, value: number }>();

  onParameterChange(paramName: keyof ScenarioParameters, value: string | number): void {
    const numericValue = Number(value);
    this.parameterChange.emit({ paramName, value: numericValue });
  }
}
