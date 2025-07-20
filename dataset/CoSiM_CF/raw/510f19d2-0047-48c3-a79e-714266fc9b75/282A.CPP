#include <bits/stdc++.h>
#define ll long long int
#define ull unsigned long long int
#define pb push_back

using namespace std;

int main() {
  std::ios_base::sync_with_stdio(false);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n, ans = 0;
  cin >> n;
  string i1 = "++X", d1 = "--X", i2 = "X++", d2 = "X--";
  for (int i = 0; i < n; i++) {
    string ip;
    cin >> ip;
    if (ip == i1 || ip == i2) {
      ans += 1;
    } else if (ip == d1 || ip == d2) {
      ans -= 1;
    }
  }
  cout << ans << "\n";
  return 0;
}
