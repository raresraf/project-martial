#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  while (cin >> s) {
    int i, j, k, l, m, n;
    l = s.length();
    i = 0, n = 0;
    for (; i < l; i++) {
      if (s[i] == 'h') {
        n++;
        i++;
        break;
      }
    }
    for (; i < l; i++) {
      if (s[i] == 'e') {
        n++;
        i++;
        break;
      }
    }
    for (; i < l; i++) {
      if (s[i] == 'l') {
        n++;
        i++;
        break;
      }
    }
    for (; i < l; i++) {
      if (s[i] == 'l') {
        n++;
        i++;
        break;
      }
    }
    for (; i < l; i++) {
      if (s[i] == 'o') {
        n++;
        i++;
        break;
      }
    }
    if (n >= 5) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
  return 0;
}
