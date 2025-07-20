#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  while (cin >> s) {
    int i, c = 0, l;
    l = s.length();
    int a[26] = {0};
    for (i = 0; i < l; i++) {
      if (s[i] == 'a') {
        a[0] = 1;
      } else if (s[i] == 'b') {
        a[1] = 1;
      } else if (s[i] == 'c') {
        a[2] = 1;
      } else if (s[i] == 'd') {
        a[3] = 1;
      } else if (s[i] == 'e') {
        a[4] = 1;
      } else if (s[i] == 'f') {
        a[5] = 1;
      } else if (s[i] == 'g') {
        a[6] = 1;
      } else if (s[i] == 'h') {
        a[7] = 1;
      } else if (s[i] == 'i') {
        a[8] = 1;
      } else if (s[i] == 'j') {
        a[9] = 1;
      } else if (s[i] == 'k') {
        a[10] = 1;
      } else if (s[i] == 'l') {
        a[11] = 1;
      } else if (s[i] == 'm') {
        a[12] = 1;
      } else if (s[i] == 'n') {
        a[13] = 1;
      } else if (s[i] == 'o') {
        a[14] = 1;
      } else if (s[i] == 'p') {
        a[15] = 1;
      } else if (s[i] == 'q') {
        a[16] = 1;
      } else if (s[i] == 'r') {
        a[17] = 1;
      } else if (s[i] == 's') {
        a[18] = 1;
      } else if (s[i] == 't') {
        a[19] = 1;
      } else if (s[i] == 'u') {
        a[20] = 1;
      } else if (s[i] == 'v') {
        a[21] = 1;
      } else if (s[i] == 'w') {
        a[22] = 1;
      } else if (s[i] == 'x') {
        a[23] = 1;
      } else if (s[i] == 'y') {
        a[24] = 1;
      } else if (s[i] == 'z') {
        a[25] = 1;
      }
    }
    for (i = 0; i < 26; i++) {
      if (a[i] == 1) {
        c++;
      }
    }
    if (c % 2 == 0) {
      cout << "CHAT WITH HER!" << endl;
    } else {
      cout << "IGNORE HIM!" << endl;
    }
  }
  return 0;
}
