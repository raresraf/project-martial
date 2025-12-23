import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ScoreDistributionChartComponent } from './score-distribution-chart.component';

describe('ScoreDistributionChartComponent', () => {
  let component: ScoreDistributionChartComponent;
  let fixture: ComponentFixture<ScoreDistributionChartComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ScoreDistributionChartComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ScoreDistributionChartComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
