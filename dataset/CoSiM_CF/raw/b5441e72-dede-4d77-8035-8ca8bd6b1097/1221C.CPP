#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  while (t--) {
    int a, b, c;
    cin >> a >> b >> c;
    int sum = a + b + c;
    sum /= 3;
    if (sum > min(a, b)) {
      sum = min(a, b);
      int x = fabs(c - sum);
      sum = min(sum, c + x);
    }
    cout << sum << endl;
  }
  return 0;
}
