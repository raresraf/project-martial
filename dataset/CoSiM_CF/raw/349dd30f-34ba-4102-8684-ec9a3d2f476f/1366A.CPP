#include <bits/stdc++.h>
using namespace std;
int main() {

  int t;
  cin >> t;
  while (t--) {
    int n, m;
    cin >> n >> m;
    int count = 0;

    while (n > 0 && m > 0) {
      if (2 * n <= m || 2 * m <= n) {
        count += min(m, n);
        if (m < n) {
          m = 0;
        } else {
          n = 0;
        }
      } else {
        int k = min(m, n) / 2;
        count += k;
        if (m <= n) {
          n -= 2 * (m / 2);
          m -= m / 2;

        } else {
          m -= 2 * (n / 2);
          n -= n / 2;
        }
      }
    }
    cout << count << endl;
  }
}