import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ResultsDisplayComponent } from './results-display.component';

describe('ResultsDisplayComponent', () => {
  let component: ResultsDisplayComponent;
  let fixture: ComponentFixture<ResultsDisplayComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ResultsDisplayComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ResultsDisplayComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
