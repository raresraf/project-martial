#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  int l = s.length();
  char cur = s[0], pre = s[0];
  int cnt = 0, ans = 0;
  for (int i = 0; i < l; i++) {
    cur = s[i];
    if (cur == pre) {
      cnt++;
    } else {
      // cout<<cnt;
      if (cnt >= 7) {
        ans = 1;
      }
      cnt = 1;
    }
    pre = cur;
  }
  if (cnt >= 7)
    ans = 1;
  if (ans == 1)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}