#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, i, s = 0, c = 0;
  cin >> n;
  int x[n];

  for (i = 0; i < n; i++) {
    cin >> x[i];
    s += x[i];
  }
  sort(x, x + n, greater<int>());

  for (i = 0; i < n & c <= s; i++) {
    c += x[i];
    s -= x[i];
  }
  cout << i;
}
