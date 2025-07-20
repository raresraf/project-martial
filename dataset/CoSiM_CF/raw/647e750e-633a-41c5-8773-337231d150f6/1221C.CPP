#include <bits/stdc++.h>
using namespace std;

void dihan() {
  int a, b, c, d, s;
  cin >> a >> b >> c;

  d = min(a, b);
  s = a + b + c;

  if (3 * d <= s)
    cout << d << endl;
  else
    cout << s / 3 << endl;

  return;
}

int main() {
  int t;
  cin >> t;
  while (t--)
    dihan();
}
