import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ParameterControlsComponent } from './parameter-controls.component';

describe('ParameterControlsComponent', () => {
  let component: ParameterControlsComponent;
  let fixture: ComponentFixture<ParameterControlsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ParameterControlsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ParameterControlsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
