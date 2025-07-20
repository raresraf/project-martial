#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string s1, s2;
  while (cin >> s1 >> s2) {
    int len1, i, value = 0, a = 0;
    len1 = s1.length();
    for (i = 0; i < len1; i++) {
      if (s1[i] >= 65 && s1[i] <= 90)
        s1[i] = s1[i] + 32;
      if (s2[i] >= 65 && s2[i] <= 90)
        s2[i] = s2[i] + 32;
      if (s1[i] < s2[i]) {
        a = 1;
        cout << "-1" << endl;
        break;
      } else if (s2[i] < s1[i]) {
        a = 1;
        cout << "1" << endl;
        break;
      }
    }
    if (a == 0)
      cout << 0 << endl;
  }
  return 0;
}
