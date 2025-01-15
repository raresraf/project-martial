// author @Nishant

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  if (!(n % 7)) {
    cout << (n / 7) * 2 << " " << (n / 7) * 2;
  } else {
    if (n % 7 >= 6) {
      cout << (2 * (n / 7)) + 1 << " " << (2 * (n / 7)) + 2;
    } else if (n % 7 <= 5 && n % 7 >= 2) {
      cout << (n / 7) * 2 << " " << (2 * (n / 7)) + 2;
    } else if (n % 7 < 2) {
      cout << (n / 7) * 2 << " " << (2 * (n / 7)) + 1;
    }
  }
  cout << endl;
  return 0;
}