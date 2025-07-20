#include <bits/stdc++.h>
using namespace std;

int main() {
  long long k, n, w, i, need, a = 0, borrow;
  cin >> k >> n >> w;
  while (w > 0) {
    a += w;
    w--;
  }
  need = k * a;
  borrow = need - n;
  if (borrow < 0)
    cout << 0 << endl;
  else
    cout << borrow << endl;
  return 0;
}
