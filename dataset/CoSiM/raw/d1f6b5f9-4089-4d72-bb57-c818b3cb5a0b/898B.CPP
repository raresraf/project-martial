#include <bits/stdc++.h>
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);
using namespace std;
long long idx = 0, x = -1, y, n, a, b;
int main() {
  optimizar_io cin >> n >> a >> b;
  for (int i = 0; i * a <= n; ++i) {
    if ((n - (i * a)) % b == 0) {
      x = i;
      y = (n - (i * a)) / b;
    }
  }
  if (x == -1)
    cout << "NO\n";
  else
    cout << "YES\n" << x << " " << y << "\n";
}
