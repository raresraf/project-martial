#include <bits/stdc++.h>
using namespace std;

int main() {
  int k;
  cin >> k;
  int n;
  cin >> n;
  int w;
  cin >> w;
  int costo_total;

  costo_total = ((w * (w + 1)) / 2) * k;
  if (costo_total <= n) {
    cout << "0";
  } else {
    cout << costo_total - n;
  }

  return 0;
}
