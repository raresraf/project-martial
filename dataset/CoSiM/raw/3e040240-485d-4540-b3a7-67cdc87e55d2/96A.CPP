#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int i, j, x = 0, c = 0;
  for (i = 0; i < s.length(); i++) {
    if (s.at(i) == '0' && s.length() > i + 6) {
      c = 0;
      for (j = i; j < i + 7; j++) {
        if (s.at(j) == '1')
          c = 1;
      }
      if (c == 0) {
        x = 1;
        break;
      }
    } else if (s.at(i) == '1' && s.length() > i + 6) {
      c = 0;
      for (j = i; j < i + 7; j++) {
        if (s.at(j) == '0')
          c = 1;
      }
      if (c == 0) {
        x = 1;
        break;
      }
    }
  }
  if (x == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}