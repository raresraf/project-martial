#include <bits/stdc++.h>
using namespace std;

int main() {
  unsigned int n, k, w;
  cin >> k >> n >> w;

  long int res = k * (w * (w + 1) / 2);

  res -= n;

  if (res <= 0)
    cout << "0\n";
  else
    cout << res << "\n";

  return 0;
}
