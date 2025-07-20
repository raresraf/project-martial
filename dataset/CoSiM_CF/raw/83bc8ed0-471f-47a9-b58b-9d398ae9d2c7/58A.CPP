#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  bool hh = false, ee = false, ll = false, oo = false;
  int h = 0, e = 0, l = 0, o = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'h') {
      h++;
      hh = true;
    }
    if (s[i] == 'e' && hh) {
      ee = true;
      e++;
    }
    if (s[i] == 'l' && hh && ee) {
      ll = true;
      l++;
    }
    if (s[i] == 'o' && hh && ee && ll) {
      oo = true;
      o++;
    }
  }
  if (hh && ee && ll && oo) {
    if (l >= 2)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  } else
    cout << "NO" << endl;
  return 0;
}
