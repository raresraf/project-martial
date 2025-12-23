import { Component, Input, OnChanges, SimpleChanges, ElementRef } from '@angular/core';
import { Chart, ChartConfiguration, ChartOptions, CategoryScale, LinearScale, LineController, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { CalculationResults } from '../../logic/model';
import { BaseChartDirective } from 'ng2-charts';
import { CommonModule } from '@angular/common';
import AnnotationPlugin from 'chartjs-plugin-annotation';

Chart.register(AnnotationPlugin, CategoryScale, LinearScale, LineController, PointElement, LineElement, Title, Tooltip, Legend);

@Component({
  selector: 'app-score-distribution-chart',
  templateUrl: './score-distribution-chart.component.html',
  styleUrls: ['./score-distribution-chart.component.css'],
  standalone: true,
  imports: [BaseChartDirective, CommonModule],
})
export class ScoreDistributionChartComponent implements OnChanges {
  @Input() results: CalculationResults | null = null;

  public data: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: 'Estimated True Positives (TP)',
        borderColor: 'rgb(15, 157, 88)',
        backgroundColor: 'rgba(15, 157, 88, 0.5)',
        tension: 0.1,
      },
      {
        data: [],
        label: 'Estimated False Positives (FP)',
        borderColor: 'rgb(219, 68, 55)',
        backgroundColor: 'rgba(219, 68, 55, 0.5)',
        tension: 0.1,
      }
    ]
  };

  public options: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Estimated TP and FP vs. Detection Threshold',
        font: {
            family: "'Roboto', 'Times New Roman', Times, serif",
            size: 18,
            weight: 'normal',
        }
      },
      annotation: {
        annotations: {}
      }
    },
    scales: {
        x: {
            title: {
                display: true,
                text: 'Detection Threshold (θ)'
            }
        },
        y: {
            title: {
                display: true,
                text: 'Estimated # of Items'
            },
            type: 'linear',
            min: 0,
        }
    }
  };

  constructor(private el: ElementRef) {}

  ngOnInit(): void {
    const computedStyle = getComputedStyle(this.el.nativeElement);
    const gcpGreen = computedStyle.getPropertyValue('--gcp-green').trim();
    const gcpRed = computedStyle.getPropertyValue('--gcp-red').trim();

    this.data.datasets[0].borderColor = gcpGreen;
    this.data.datasets[0].backgroundColor = this.hexToRgba(gcpGreen, 0.5);
    this.data.datasets[1].borderColor = gcpRed;
    this.data.datasets[1].backgroundColor = this.hexToRgba(gcpRed, 0.5);
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['results'] && this.results) {
      this.updateChart();
    }
  }

  private updateChart(): void {
    if (!this.results) return;

    const newLabels: (string | null)[] = [];
    const newTPData: (number | null)[] = [];
    const newFPData: (number | null)[] = [];

    const gapThreshold = 0.2;

    for (let i = 0; i < this.results.tilde_TP_values.length; i++) {
      const currentPoint = this.results.tilde_TP_values[i];
      const currentFPPoint = this.results.tilde_FP_values[i];

      if (i > 0) {
        const prevPoint = this.results.tilde_TP_values[i - 1];
        if (currentPoint.theta - prevPoint.theta > gapThreshold) {
          newLabels.push('...');
          newTPData.push(null);
          newFPData.push(null);
        }
      }

      newLabels.push(currentPoint.theta.toFixed(2));
      newTPData.push(currentPoint.value);
      newFPData.push(currentFPPoint.value);
    }
    
    this.data = {
      labels: newLabels as string[],
      datasets: [
        { ...this.data.datasets[0], data: newTPData },
        { ...this.data.datasets[1], data: newFPData }
      ]
    };

    
  }

  private hexToRgba(hex: string, alpha: number): string {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);

    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
}
