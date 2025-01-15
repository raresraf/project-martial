#include <bits/stdc++.h>
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

int main() {
  _ int n = 5, e;
  int nrow = 0, ncol = 0;
  for (int i = 0; i < n * n; i++) {
    cin >> e;
    if (e == 1) {
      nrow = i / n;
      ncol = i % n;
      break;
    }
  }
  cout << abs(nrow - 2) + abs(ncol - 2) << "\n";

  return 0;
}