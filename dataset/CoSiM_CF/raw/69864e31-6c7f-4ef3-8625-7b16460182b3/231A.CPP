#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int a[n][3], sum, count = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }
  for (int i = 0; i < n; i++) {
    sum = 0;
    for (int j = 0; j < 3; j++) {
      sum += a[i][j];
    }
    if (sum > 1)
      count += 1;
  }
  cout << count;
  // your code goes here

  return 0;
}
