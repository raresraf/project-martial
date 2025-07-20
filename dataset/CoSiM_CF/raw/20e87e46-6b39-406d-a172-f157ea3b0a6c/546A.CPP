#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int k, n, w, m = 0, i = 1;

  cin >> k >> n >> w;

  while (w--) {
    m += k * i;
    ++i;
  }

  if (n - m < 0) {
    cout << (-1) * (n - m);
  } else {
    cout << "0";
  }

  return 0;
}
