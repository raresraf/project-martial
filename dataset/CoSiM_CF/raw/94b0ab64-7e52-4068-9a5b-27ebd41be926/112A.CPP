#include <bits/stdc++.h>

using namespace std;

int main() {
  string a, b;

  cin >> a >> b;

  int len = a.length();

  for (int i = 0; i < len; i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }

  if (a.compare(b) > 0)
    cout << 1 << endl;
  else if (a.compare(b) < 0)
    cout << -1 << endl;
  else
    cout << 0 << endl;

  return 0;
}
