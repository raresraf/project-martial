import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkDataEditorComponent } from './benchmark-data-editor.component';

describe('BenchmarkDataEditorComponent', () => {
  let component: BenchmarkDataEditorComponent;
  let fixture: ComponentFixture<BenchmarkDataEditorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [BenchmarkDataEditorComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkDataEditorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
