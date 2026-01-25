#include <bits/stdc++.h>
using namespace std;
main() {
  int k, n, w;
  cin >> k >> n >> w;

  int total = ((w * (w + 1)) / 2) * k;

  if ((total - n) > 0)
    cout << total - n;
  else
    cout << "0";
}
