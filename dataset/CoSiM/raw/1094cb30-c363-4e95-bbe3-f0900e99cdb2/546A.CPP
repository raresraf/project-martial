// author @Nishant

#include <bits/stdc++.h>
using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int i = 1;
  while (n > 0 && w >= 0) {
    if (n - i * k > 0) {
      n -= i * k;
      i++;
      w--;
    } else {
      break;
    }
  }
  int needed = 0;
  while (w > 0) {
    needed += i * k;
    w--;
    i++;
  }
  if (needed > 0) {
    cout << needed - n;
  } else {
    cout << 0;
  }
  return 0;
}