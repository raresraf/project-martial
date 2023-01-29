import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { DiffComponent } from './diff/diff.component';

const routes: Routes = [
  {path: '', redirectTo: '/', pathMatch: 'full'},
  {path: 'diff', component: DiffComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
