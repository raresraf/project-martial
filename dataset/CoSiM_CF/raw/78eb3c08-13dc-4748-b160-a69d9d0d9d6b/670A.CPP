#include <bits/stdc++.h>
#define lli long long int
using namespace std;

int main() {
  lli n;
  cin >> n;
  lli res = n / 7;
  lli rem = n % 7;
  if (n < 7) {
    if (n >= 2) {
      if (n < 6)
        cout << 0 << ' ' << 2 << endl;
      else
        cout << 1 << ' ' << 2 << endl;
    } else {
      cout << 0 << ' ' << 1 << endl;
    }
  } else {
    res *= 2;
    if (rem == 0) {
      cout << res << ' ' << res << endl;
    } else if (rem >= 2) {
      if (rem < 6)
        cout << res << ' ' << res + 2 << endl;
      else
        cout << res + 1 << ' ' << res + 2 << endl;
    } else {
      cout << res << ' ' << res + 1 << endl;
    }
  }

  return 0;
}
