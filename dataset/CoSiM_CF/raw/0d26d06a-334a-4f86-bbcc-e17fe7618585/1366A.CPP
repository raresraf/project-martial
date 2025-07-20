#include <bits/stdc++.h>
using namespace std;
int main() {
  int t;
  cin >> t;
  while (t--) {
    long long int ans;
    int a, b;
    cin >> a >> b;
    if (a == 0 || b == 0 || (a == 1 && b == 1))
      ans = 0;
    else if ((a == 1 && b >= 2) || (b == 1 && a >= 2))
      ans = 1;
    else if (a >= 2 * b)
      ans = b;
    else if (b >= 2 * a)
      ans = a;
    else {
      ans = a / 3 + b / 3;
      int sum = a % 3 + b % 3;
      if (sum >= 3)
        ans++;
    }
    cout << ans << endl;
  }
}
