#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, a, b, c, ans = 0;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a >> b >> c;
    if (a + b + c >= 2) {
      ans++;
    }
  }
  cout << ans << endl;
  return 0;
}
