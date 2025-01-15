#include <bits/stdc++.h>

using namespace std;

int main() {

  int a[105], n = 0, i;

  while (cin >> a[n++])
    ;
  sort(a, a + --n);
  cout << a[0];
  for (i = 1; i < n; i++) {
    cout << '+' << a[i];
  }
}