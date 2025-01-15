#include <bits/stdc++.h>
using namespace std;
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);

typedef long long int ll;

int main() {
  optimizar_io int tc, n;

  cin >> tc;
  while (tc--) {
    cin >> n;
    int ans = numeric_limits<int>::max();
    for (int i = 1; i * i <= n; i++) {
      int rem = n % i;
      int steps = i - 1;
      steps += (n / i) - 1;
      if (rem)
        steps++;
      ans = min(ans, steps);
    }
    cout << ans << "\n";
  }
  return 0;
}
