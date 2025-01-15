/*                     */

#include <bits/stdc++.h>
using namespace std;

int main() {
  char a[52];
  int n, i, f = 0;

  cin >> n;

  for (i = 0; i < n; i++) {
    cin >> a[i];
  }
  for (i = 0; i < n; i++) {
    if (a[i] == a[i + 1])
      f++;
  }
  cout << f;

  return 0;
}
