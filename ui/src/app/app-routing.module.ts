import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { DiffComponent } from './diff/diff.component';
import { ThresholdAppComponent } from './threshold-app/threshold-app.component';
import { LeaderboardComponent } from './leaderboard/leaderboard.component';
import { NetworkTrafficDatasetComponent } from './network-traffic-dataset/network-traffic-dataset.component';
import { CosimDatasetComponent } from './cosim-dataset/cosim-dataset.component';

const routes: Routes = [
  {path: 'similarity', component: DiffComponent},
  {path: 'threshold-ops', component: ThresholdAppComponent},
  {path: 'leaderboard', component: LeaderboardComponent},
  {path: 'network-traffic-dataset', component: NetworkTrafficDatasetComponent},
  {path: 'cosim-dataset', component: CosimDatasetComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
