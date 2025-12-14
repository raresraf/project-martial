#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, a, b, ca = 0, ct = 0;
  cin >> t;
  while (t--) {
    cin >> a >> b;
    ca += b;
    ca -= a;
    if (ca >= ct) {
      ct = ca;
    }
  }
  cout << ct << endl;
  return 0;
}
