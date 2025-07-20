#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int k, n, w, r;
  cin >> k >> n >> w;
  r = k * w * (w + 1) * 0.5;
  if (n >= r)
    cout << 0;
  else
    cout << r - n;
  return 0;
}
