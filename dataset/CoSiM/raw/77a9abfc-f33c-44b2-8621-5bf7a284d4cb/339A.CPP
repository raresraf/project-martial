#include <bits/stdc++.h>
using namespace std;

int main() {
  // your code goes here
  string s;
  cin >> s;
  int c1 = 0, c2 = 0, c3 = 0;

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
  while (c1 != 0) {
    cout << "1";
    c1--;
    if (c1 != 0 || c2 != 0 || c3 != 0)
      cout << "+";
  }
  while (c2 != 0) {
    cout << "2";
    c2--;
    if (c2 != 0 || c3 != 0)
      cout << "+";
  }
  while (c3 != 0) {
    cout << "3";
    c3--;
    if (c3)
      cout << "+";
  }
  return 0;
}
