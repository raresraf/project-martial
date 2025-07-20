#include <bits/stdc++.h>
using namespace std;

int main() {
  int c1, c2, c3;
  c1 = c2 = c3 = 0;
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      c1++;
    }
    if (s[i] == '2') {
      c2++;
    }
    if (s[i] == '3') {
      c3++;
    }
  }
  for (int i = 0; i < c1; i++) {
    cout << 1;
    if (c1 - i == 1 && c2 == 0 && c3 == 0) {
      return 0;
    } else {
      cout << "+";
    }
  }
  for (int i = 0; i < c2; i++) {
    cout << 2;
    if (c2 - i == 1 && c3 == 0) {
      return 0;
    } else {
      cout << "+";
    }
  }
  for (int i = 0; i < c3; i++) {
    cout << 3;
    if (c3 - i == 1) {
      return 0;
    } else {
      cout << "+";
    }
  }
  return 0;
}