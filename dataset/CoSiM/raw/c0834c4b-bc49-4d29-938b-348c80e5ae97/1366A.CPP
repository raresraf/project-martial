#include <bits/stdc++.h>
using namespace std;
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  long long int t, i, a, b, x;
  cin >> t;
  for (i = 1; i <= t; i++) {
    cin >> a >> b;
    x = min(a, b);
    x = min(x, (a + b) / 3);
    cout << x << "\n";
  }
}