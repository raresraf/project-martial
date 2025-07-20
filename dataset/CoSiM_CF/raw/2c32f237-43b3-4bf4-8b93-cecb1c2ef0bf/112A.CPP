// http://codeforces.com/problemset/problem/112/A

#include <iostream>
using namespace std;

string cvt(string a) {
  int sz = a.length();
  for (int i = 0; i < sz; i++) {
    if (a[i] >= 'a')
      a[i] -= 32;
  }
  return a;
}

int main() {
  string a, b;
  cin >> a >> b;

  a = cvt(a);
  b = cvt(b);

  if (a == b)
    cout << 0;
  else if (a < b)
    cout << -1;
  else
    cout << 1;

  return 0;
}
