#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n, count = 0, sum = 0;
  cin >> n;
  int arr[n][3];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      sum += arr[i][j];
    }
    if (sum > 1) {
      count++;
    }
    sum = 0;
  }
  cout << count << endl;
}