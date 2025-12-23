import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PrecisionRecallChartComponent } from './precision-recall-chart.component';

describe('PrecisionRecallChartComponent', () => {
  let component: PrecisionRecallChartComponent;
  let fixture: ComponentFixture<PrecisionRecallChartComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [PrecisionRecallChartComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PrecisionRecallChartComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
