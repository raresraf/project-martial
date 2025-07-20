#include <bits/stdc++.h>
using namespace std;
main() {
  char s[200];
  int i, j, n, m;
  while (cin >> s) {
    n = strlen(s);
    m = 0;
    for (i = 0; i < n; i++) {
      if (s[i] == 'h') {
        if (s[i + 1] == 'h' || s[i + 1] == 'e')
          m = 1;
        else {
          m = 0;
          break;
        }
      }
      if (s[i] == 'e') {
        if (s[i + 1] == 'e' || s[i + 1] == 'l')
          m = 1;
        else {
          m = 0;
          break;
        }
      }
      if (s[i] == 'l') {
        if (s[i + 1] == 'l' || s[i + 1] == 'o')
          m = 1;
        else {
          m = 0;
          break;
        }
      }
      if (s[i] == 'o') {
        m = 1;
      }
    }
    if (m == 1)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
}
