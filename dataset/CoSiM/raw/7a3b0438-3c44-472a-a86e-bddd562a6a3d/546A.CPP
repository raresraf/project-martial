#include <bits/stdc++.h>

using namespace std;

int main() {
  int k, n, w, amount = 0;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++) {
    amount += k * i;
  }
  if (amount <= n)
    cout << 0 << endl;
  else
    cout << amount - n << endl;
  return 0;
}
