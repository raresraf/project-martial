#include <bits/stdc++.h>
using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  for (int i = 0; a[i] != '\0'; i++) {
    if (a[i] >= 'A' && a[i] <= 'Z') {
      a[i] = char(int(a[i]) + 32);
    }
  }

  for (int i = 0; b[i] != '\0'; i++) {
    if (b[i] >= 'A' && b[i] <= 'Z') {
      b[i] = char(int(b[i]) + 32);
    }
  }
  if (a == b)
    cout << 0 << endl;
  else if (a > b)
    cout << 1 << endl;
  else
    cout << -1 << endl;
  return 0;
}
