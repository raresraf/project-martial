#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, i, j, c = 0, maxi = 0;
  cin >> n;
  int a[n], b[n];
  for (i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
    maxi = maxi - a[i] + b[i];
    if (maxi > c)
      c = maxi;
  }
  cout << c << endl;
  return 0;
}
