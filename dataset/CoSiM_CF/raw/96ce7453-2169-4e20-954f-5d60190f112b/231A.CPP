#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  int ans = 0;
  while (t--) {
    int a, b, c;
    cin >> a >> b >> c;
    int sum = a + b + c;
    if (sum > 1)
      ans++;
  }
  cout << ans;
}
