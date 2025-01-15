#include <bits/stdc++.h>
using namespace std;
main() {
  char s[120];
  int i, n, l, j, flag;
  while (cin >> s) {
    l = strlen(s);
    for (i = 0; i < l; i++) {
      if (s[i] != '1') {
        for (j = i + 1; j < l; j++) {
          if (s[i] != s[j]) {
            flag = 0;
          } else {
            // flag=1;
            s[j] = '1';
          }
        }
      }
    }
    int c = 0;
    for (i = 0; i < l; i++) {
      if (s[i] != '1') {
        c++;
      }
    }
    if (c % 2 == 0) {
      cout << "CHAT WITH HER!" << endl;
    } else
      cout << "IGNORE HIM!" << endl;
  }
}
