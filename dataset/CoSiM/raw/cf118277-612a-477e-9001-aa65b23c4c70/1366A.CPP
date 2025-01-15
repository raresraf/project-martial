#include <bits/stdc++.h>
using namespace std;
int main() {
  int t;
  cin >> t;
  while (t > 0) {
    int a, b;
    cin >> a >> b;
    if (a == 0 || b == 0)
      cout << "0" << endl;
    else if (a >= 2 * b)
      cout << b << endl;
    else if (b >= 2 * a)
      cout << a << endl;
    else {
      int Sw = (2 * b - a) / 3;
      int Sh = 0;
      if (Sw < 0) {
        Sw = 0;
        Sh = a / 2;
      } else {
        Sh = (a - Sw) / 2;
        if (Sh < 0)
          Sh = 0;
      }

      cout << Sw + Sh << endl;
    }
    t--;
  }
}