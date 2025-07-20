#include <bits/stdc++.h>
using namespace std;
#define pb push_back
#define pf push_front
#define ll long long

int main() {
  // freopen("input.txt", "r", stdin);
  // freopen("output.txt", "w", stdout);

  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int t;
  cin >> t;
  while (t--) {
    int c, m, x;
    cin >> c >> m >> x;
    int ans = min(min(c, m), (c + m + x) / 3);

    cout << ans << "\n";
  }

  return 0;
}