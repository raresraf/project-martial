#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int i, j, k, l, m, n, o;
  while (cin >> s) {
    l = s.length();
    o = 0;
    string s1;
    for (i = 0; i < l; i++) {
      if (s[i] != '+') {
        s1 += s[i];
      } else {
        o = 1;
      }
    }
    k = s1.length();
    sort(s1.begin(), s1.end());
    if (o == 0) {
      for (i = 0; i < k; i++) {
        cout << s1[i];
      }
      cout << endl;
    } else {
      for (i = 0; i < k; i++) {
        cout << s1[i];
        if (i < k - 1) {
          cout << '+';
        }
      }
      cout << endl;
    }
  }
  return 0;
}
