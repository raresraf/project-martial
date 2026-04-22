import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { DiffComponent } from './diff/diff.component';
import { ThresholdAppComponent } from './threshold-app/threshold-app.component';
import { LeaderboardComponent } from './leaderboard/leaderboard.component';

const routes: Routes = [
  {path: 'diff', component: DiffComponent},
  {path: 'threshold-ops', component: ThresholdAppComponent},
  {path: 'leaderboard', component: LeaderboardComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
