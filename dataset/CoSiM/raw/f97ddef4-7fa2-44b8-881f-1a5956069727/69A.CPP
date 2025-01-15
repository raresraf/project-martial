#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, j, k, isum(0), jsum(0), ksum(0);
  cin >> n;
  for (int it = 0; it < n; it++) {
    cin >> i >> j >> k;
    isum += i;
    jsum += j;
    ksum += k;
  }
  if (isum == 0 && jsum == 0 && ksum == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
