#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  int n = s.length();
  if (n < 7)
    cout << "NO";
  else if (n >= 7) {
    int cnt = 1, f;
    if (s[0] == '0')
      f = 0;
    else
      f = 1;
    int i = 1;
    while (cnt != 7 && i < n) {
      if (s[i] == '1' && f == 1) {
        cnt++;
      } else if (s[i] == '1' && f == 0) {
        cnt = 0;
        cnt++;
        f = 1;
      } else if (s[i] == '0' && f == 0) {
        cnt++;
      } else {
        cnt = 0;
        cnt++;
        f = 0;
      }
      i++;
    }
    if (cnt == 7)
      cout << "YES";
    else
      cout << "NO";
  }
}
