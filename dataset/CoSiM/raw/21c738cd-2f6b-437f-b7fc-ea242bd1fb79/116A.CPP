#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, min = 0;
  cin >> n;
  int out[n], in[n];
  for (int i = 0; i < n; i++) {
    cin >> out[i] >> in[i];
  }
  int x = in[0];
  for (int i = 0; i < n - 1; i++) {
    min += in[i];
    if (min >= x)
      x = min;
    min = min - out[i + 1];
  }
  cout << x;
  return 0;
}