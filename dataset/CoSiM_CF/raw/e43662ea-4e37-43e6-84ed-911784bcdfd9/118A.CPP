#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i;
  string a;
  cin >> a;
  n = a.length();
  for (i = 0; i < n; i++) {
    if (a[i] >= 65 && a[i] <= 90)
      a[i] = a[i] - 65 + 'a';
    if (a[i] == 'a' || a[i] == 'e' || a[i] == 'i' || a[i] == 'o' ||
        a[i] == 'u' || a[i] == 'y')
      continue;
    else
      cout << "." << a[i];
  }
  return 0;
}
