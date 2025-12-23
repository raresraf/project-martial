import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { DiffComponent } from './diff/diff.component';
import { ThresholdAppComponent } from './threshold-app/threshold-app.component';

const routes: Routes = [
  {path: '', redirectTo: '/', pathMatch: 'full'},
  {path: 'diff', component: DiffComponent},
  {path: 'threshold-ops', component: ThresholdAppComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
