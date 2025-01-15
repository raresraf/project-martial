#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  int a = 0;
  if (int(s[0]) - 'a' <= 26 && int(s[0]) - 'a' >= 0) {
    a = int(s[0]);
    a -= 32;
    s[0] = (char)a;
  }
  cout << s << endl;
  return 0;
}
