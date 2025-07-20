#include <bits/stdc++.h>
using namespace std;
#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);

int main() {
  fast int n, a, b, ans = 0, x;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    x = b - a;
    if (x > 0)
      ans += x;
  }
  cout << ans << endl;
  return 0;
}
