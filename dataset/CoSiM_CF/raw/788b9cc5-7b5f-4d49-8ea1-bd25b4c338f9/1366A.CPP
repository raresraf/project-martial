#include <bits/stdc++.h>
using namespace std;

int main() {

  int t;
  cin >> t;
  while (t--) {
    int a, b, ans = 0;
    cin >> a >> b;
    ans = min(min(a, b), (a + b) / 3);
    cout << ans << endl;
  }

  return 0;
}
