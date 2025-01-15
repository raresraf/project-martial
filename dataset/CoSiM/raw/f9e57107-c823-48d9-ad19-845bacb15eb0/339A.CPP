#include <bits/stdc++.h>
using namespace std;
int main() {
  ios_base::sync_with_stdio(0);
  string s;
  cin >> s;
  int a = 0, b = 0, c = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1')
      a++;
    if (s[i] == '2')
      b++;
    if (s[i] == '3')
      c++;
  }
  for (int i = 0; i < a; i++) {
    if (i == a - 1)
      cout << "1";
    else
      cout << "1+";
  }
  if (b != 0) {
    if (a != 0)
      cout << "+";
    for (int i = 0; i < b; i++) {
      if (i == b - 1)
        cout << "2";
      else
        cout << "2+";
    }
  }
  if (c != 0) {
    if (a != 0 || b != 0)
      cout << "+";
    for (int i = 0; i < c; i++) {
      if (i == c - 1)
        cout << "3";
      else
        cout << "3+";
    }
  }

  return 0;
}
