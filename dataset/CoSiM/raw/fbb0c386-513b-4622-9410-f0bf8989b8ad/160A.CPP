#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, i, j, s1 = 0, s2;
  cin >> n;
  int a[n];
  for (i = 0; i < n; i++) {
    cin >> a[i];
  }
  sort(a, a + n);
  for (i = n - 1; i >= 0; i--) {
    s1 = s1 + a[i];
    s2 = 0;
    for (j = 0; j < i; j++) {
      s2 = s2 + a[j];
    }
    if (s1 > s2) {
      cout << n - i;
      return 0;
    }
  }

  return 0;
}
