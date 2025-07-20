#include <bits/stdc++.h>
using namespace std;
int main() {
  int v = 0, c = 0, l, con = 0, t;
  string a, b;
  cin >> a;
  l = a.length();
  for (int i = 0; i < l; i++) {
    if (a[i] != 'A' && a[i] != 'E' && a[i] != 'I' && a[i] != 'O' &&
        a[i] != 'U' && a[i] != 'Y' && a[i] != 'a' && a[i] != 'e' &&
        a[i] != 'i' && a[i] != 'o' && a[i] != 'u' && a[i] != 'y') {
      if (a[i] < 97) {
        a[i] = a[i] + 32;
        cout << "." << a[i];
      } else {
        cout << "." << a[i];
      }
    }
    c++;
  }
  return 0;
}
