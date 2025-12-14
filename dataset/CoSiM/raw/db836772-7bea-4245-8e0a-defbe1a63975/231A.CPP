#include <bits/stdc++.h>
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

int main() {
  _ int n;
  cin >> n;
  int ans = 0;
  while (n--) {
    int x = 0, a, b, c;
    cin >> a >> b >> c;
    x = a + b + c;
    if (x > 1)
      ans++;
  }
  cout << ans << "\n";

  return 0;
}