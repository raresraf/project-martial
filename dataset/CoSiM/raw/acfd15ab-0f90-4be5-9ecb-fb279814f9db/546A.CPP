#include <bits/stdc++.h>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  long long k, w, n, m;
  cin >> k >> n >> w;
  m = k * ((w * (w + 1)) / 2);
  if (m > n)
    cout << m - n << endl;
  else
    cout << 0 << endl;
  return 0;
}
