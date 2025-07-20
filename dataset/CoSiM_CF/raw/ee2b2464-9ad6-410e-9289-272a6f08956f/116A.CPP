#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, a[100000], b[100000], c[100000], max;
  cin >> n;
  for (i = 0; i < n; i++)
    cin >> a[i] >> b[i];
  c[-1] = 0;
  for (i = 0; i < n; i++) {
    c[i] = b[i] - a[i] + c[i - 1];
  }
  max = c[0];
  for (i = 0; i < n; i++) {
    if (c[i] > max) {
      max = c[i];
    }
  }
  cout << max;
  return 0;
}
