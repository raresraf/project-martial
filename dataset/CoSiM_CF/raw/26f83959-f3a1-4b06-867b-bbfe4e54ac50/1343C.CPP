#include <bits/stdc++.h>
using namespace std;
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);

typedef long long int ll;

ll a[200010];

int main() {
  optimizar_io int tc, n;

  cin >> tc;
  while (tc--) {
    cin >> n;
    for (int i = 0; i < n; i++)
      cin >> a[i];
    int prev = (a[0] > 0 ? 1 : 0), len = 1;
    ll sum = 0, mx = a[0];
    for (int i = 1; i < n; i++) {
      if (prev) { // 1 - Positivo
        if (a[i] < 0) {
          prev = 1 - prev;
          len++;
          sum += mx;
          mx = a[i];
        } else {
          mx = max(mx, a[i]);
        }
      } else { // 0 - Negativo
        if (a[i] > 0) {
          prev = 1 - prev;
          len++;
          sum += mx;
          mx = a[i];
        } else {
          mx = max(mx, a[i]);
        }
      }
    }
    sum += mx;
    cout << sum << "\n";
  }
  return 0;
}
