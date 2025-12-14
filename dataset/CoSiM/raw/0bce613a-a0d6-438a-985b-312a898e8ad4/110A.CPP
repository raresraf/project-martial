#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  while (cin >> s) {
    int i, j, k, l;
    l = s.length();
    k = 0;
    for (i = 0; i < l; i++) {
      if (s[i] == '4' || s[i] == '7') {
        k++;
      }
    }
    if (k == 4 || k == 7) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
  return 0;
}
