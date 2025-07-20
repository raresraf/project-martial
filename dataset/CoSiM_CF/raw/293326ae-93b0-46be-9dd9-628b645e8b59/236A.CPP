#include <bits/stdc++.h>
using namespace std;
char b[10010];

int main() {
  int n, m, i, j, q, y, aa, bb, s = 0, k, min = 1000000010, r, t;
  char a[10010];
  cin >> a;

  n = strlen(a);
  // cout << n;
  for (i = 0; i < n; i++) {
    m = a[i];
    // cout << m;
    b[m - 97] = 1;
  }
  for (i = 0; i < 26; i++) {
    if (b[i] == 1)
      s++;
  }
  if (s % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}