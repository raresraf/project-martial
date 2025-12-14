// http://codeforces.com/problemset/problem/96/A

#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;
  int cnt = 0, sz = s.length();
  bool flag = false;
  for (int i = 0; i < sz; i++) {
    if (s[i] == '0')
      cnt++;
    else {
      cnt = 0;
    }
    if (cnt >= 7)
      flag = true;
  }
  cnt = 0;
  for (int i = 0; i < sz; i++) {
    if (s[i] == '1')
      cnt++;
    else {
      cnt = 0;
    }
    if (cnt >= 7)
      flag = true;
  }
  if (flag)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
