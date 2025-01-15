#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  string res;
  for (int i = 0; i < s.length(); i++) {
    if (i == 0)
      res = res + char(toupper(s[i]));
    else
      res = res + s[i];
  }
  cout << res;
}
