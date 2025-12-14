#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

int main() {
  ll n, mx, mn;
  mx = mn = 0;
  scanf("%lld", &n);
  for (int i = 1; i <= n; ++i) {
    if ((i - 1) % 7 == 0 || (i - 2) % 7 == 0)
      mx++;
    if (i % 7 == 0 || (i + 1) % 7 == 0)
      mn++;
  }
  cout << mn << " " << mx << endl;
  return 0;
}

/*
Seg Ter Quar Quint Sex Sab Dom
                        1   2
3    4    5    6    7   8   9
10  11   12    13   14  15  16
*/