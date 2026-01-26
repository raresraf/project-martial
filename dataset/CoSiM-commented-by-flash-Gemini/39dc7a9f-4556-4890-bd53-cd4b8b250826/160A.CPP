#include <bits/stdc++.h>
using namespace std;

int s1, s2;

int main() {
  int n;
  cin >> n;
  int a[1000];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    s1 += a[i];
  }

  sort(a, a + n);
  for (int i = n - 1; i >= 0; i--) {
    s2 += a[i];
    s1 -= a[i];
    if (s2 > s1) {
      cout << n - i;
      break;
    }
  }
  return 0;
}