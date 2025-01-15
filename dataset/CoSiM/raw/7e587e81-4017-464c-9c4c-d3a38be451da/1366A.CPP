#include <bits/stdc++.h>
using namespace std;
int main() {
  int t, x = 0;
  cin >> t;
  while (t--) {
    long long int a, b, x, y;
    cin >> a >> b;
    x = min(a, ((a + b) / 3));
    y = min(b, x);
    cout << y << endl;
  }
  return 0;
}
