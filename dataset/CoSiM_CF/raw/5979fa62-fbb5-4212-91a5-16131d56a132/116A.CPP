#include <bits/stdc++.h>
using namespace std;
int main() {
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n, sum = 0, max = 0;
  cin >> n;
  int arr[n][2];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++) {
      cin >> arr[i][j];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++) {
      if (j == 0) {
        sum = sum - arr[i][j];
      } else if (j == 1) {
        sum = sum + arr[i][j];
      }
    }
    if (max < sum) {
      max = sum;
    }
  }
  cout << max;
  return 0;
}