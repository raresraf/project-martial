#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
  ll k, n, w;
  cin >> k >> n >> w;
  ll ok = (w * (w + 1)) / 2;
  ll res = ok * k;
  if (n < res)
    cout << res - n << endl;
  else
    cout << "0" << endl;
}
