#include <bits/stdc++.h>
using namespace std;
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  long long n, a, b;
  cin >> n >> a >> b;
  if (n % a == 0) {
    cout << "YES" << endl;
    cout << n / a << " "
         << "0" << endl;
    return 0;
  }
  if (n % b == 0) {
    cout << "YES" << endl;
    cout << "0"
         << " " << n / b << endl;
    return 0;
  }

  for (int i = 1; i <= n; ++i) {
    if ((n - a * i) >= b) {
      if ((n - a * i) % b == 0) {
        cout << "YES" << endl;
        cout << i << " " << (n - a * i) / b << endl;
        return 0;
      }
    } else
      break;
  }
  cout << "NO" << endl;
}
