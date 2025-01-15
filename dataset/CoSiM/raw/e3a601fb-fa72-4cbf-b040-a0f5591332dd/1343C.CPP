#include <bits/stdc++.h>
using namespace std;
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  long long int t, i, a, j, flag, maxx, sum1, sum2, cou1, cou2;
  cin >> t;
  for (i = 1; i <= t; i++) {
    sum1 = 0;
    cin >> a;
    flag = 0;
    sum2 = 0;
    maxx = -1e16;
    cou1 = 0;
    cou2 = 0;
    long long int ara[a];
    for (j = 0; j < a; j++) {
      cin >> ara[j];
    }
    for (j = 0; j < a; j++) {
      if (ara[j] > 0 && flag == 0) {
        flag = 1;
        maxx = ara[j];
      } else if (flag == 1 && ara[j] > 0) {
        maxx = max(maxx, ara[j]);
      } else if (flag == 1 && ara[j] < 0) {
        sum1 += maxx;
        maxx = ara[j];
        flag = 2;
        cou1++;
      } else if (flag == 2 && ara[j] < 0) {
        maxx = max(maxx, ara[j]);
      } else if (flag == 2 && ara[j] > 0) {
        flag = 1;
        sum1 += maxx;
        maxx = ara[j];
        cou1++;
      }
    }
    sum1 += maxx;
    cou1++;
    flag = 0;
    maxx = -1e18;
    for (j = 0; j < a; j++) {
      if (ara[j] < 0 && flag == 0) {
        flag = 1;
        maxx = ara[j];
      } else if (flag == 1 && ara[j] < 0) {
        maxx = max(maxx, ara[j]);
      } else if (flag == 1 && ara[j] > 0) {
        sum2 += maxx;
        maxx = ara[j];
        flag = 2;
        cou2++;
      } else if (flag == 2 && ara[j] > 0) {
        maxx = max(maxx, ara[j]);
      } else if (flag == 2 && ara[j] < 0) {
        flag = 1;
        sum2 += maxx;
        maxx = ara[j];
        cou2++;
      }
      // cout<<maxx<<endl;
    }
    sum2 += maxx;
    cou2++;
    if (cou1 > cou2) {
      cout << sum1 << "\n";
    } else if (cou1 < cou2) {
      cout << sum2 << "\n";
    } else {
      cout << max(sum1, sum2) << "\n";
    }
  }
}