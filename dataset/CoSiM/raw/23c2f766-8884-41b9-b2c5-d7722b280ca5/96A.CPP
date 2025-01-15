#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int i, c, d, j, x;
  while (cin >> s) {

    for (i = 0; i < s.size(); i++) {
      c = 0;
      d = 0;
      j = i;
      if (s[j] == '0') {
        while (s[j] == '0') {
          c++;
          j++;
        }

      }

      else if (s[j] == '1') {
        while (s[j] == '1') {
          d++;
          j++;
        }
      }
      if (c >= 7 || d >= 7) {
        x = 1;
        break;
      } else
        x = 0;
    }
    if (x == 1)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }

  return 0;
}
