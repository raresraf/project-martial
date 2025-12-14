#include <bits/stdc++.h>
using namespace std;
int main() {
  int k, n, w;
  cin >> k >> n >> w;
  if ((((w * (w + 1)) / 2) * k) - n >= 0)
    cout << (((w * (w + 1)) / 2) * k) - n;
  else
    cout << "0";

  return 0;
}
