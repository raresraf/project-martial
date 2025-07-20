#include <bits/stdc++.h>

using namespace std;

#define ll long long int

int main() {
  ll k, n, w;

  cin >> k >> n >> w;

  w = (w * (w + 1)) / 2;
  w = w * k;

  if (w - n > 0)
    cout << w - n << endl;
  else
    cout << 0 << endl;

  return 0;
}
