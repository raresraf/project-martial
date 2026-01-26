#include <bits/stdc++.h>
#define ll long long
using namespace std;
int main() {
  ll a, b, t;
  cin >> t;
  while (t--) {
    cin >> a >> b;
    if (a == 0 || b == 0)
      cout << "0" << endl;
    else if (2 * a < b)
      cout << a << endl;
    else if (2 * b < a)
      cout << b << endl;
    else
      cout << (a + b) / 3 << endl;
  }
}
