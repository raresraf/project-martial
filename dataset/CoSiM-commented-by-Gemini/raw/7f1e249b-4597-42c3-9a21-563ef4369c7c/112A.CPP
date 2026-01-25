
#include <bits/stdc++.h>

using namespace std;

string a, b;
string c, d;

int main() {
  cin >> a >> b;
  int n = (int)(a.length());
  for (int i = 0; i < n; i++) {
    if ('A' <= a[i] && a[i] <= 'Z')
      c.push_back(a[i] - 'A' + 'a');
    else
      c.push_back(a[i]);
    if ('A' <= b[i] && b[i] <= 'Z')
      d.push_back(b[i] - 'A' + 'a');
    else
      d.push_back(b[i]);
  }
  if (c < d)
    cout << -1;
  if (c == d)
    cout << 0;
  if (c > d)
    cout << 1;
}
