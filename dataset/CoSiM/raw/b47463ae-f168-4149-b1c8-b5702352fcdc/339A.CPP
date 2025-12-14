#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  string s;
  getline(cin, s);
  int c1 = 0, c2 = 0, c3 = 0;
  if (s.length() < 3)
    cout << s[0];
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1')
      c1++;
    if (s[i] == '2')
      c2++;
    if (s[i] == '3')
      c3++;
  }
  if (c1 > 0) {
    string res = "1";
    for (int i = 1; i < c1; i++)
      res = res + "+1";
    for (int i = 0; i < c2; i++)
      res = res + "+2";
    for (int i = 0; i < c3; i++)
      res = res + "+3";
    cout << res;
  } else if (c2 > 0) {
    string res = "2";
    for (int i = 1; i < c2; i++)
      res = res + "+2";
    for (int i = 0; i < c3; i++)
      res = res + "+3";
    cout << res;
  } else {
    string res = "3";
    for (int i = 1; i < c3; i++)
      res = res + "+3";
    cout << res;
  }

  return 0;
}
