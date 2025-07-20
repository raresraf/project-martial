#include <bits/stdc++.h>

using namespace std;

int main()

{
  int n;
  cin >> n;
  int m;
  int a[n];
  int p = 0;
  int s = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    p = p + a[i];
  }
  sort(a, a + n);
  for (int i = n - 1; i >= 0; i--) {
    s = s + a[i];
    if (s > p - s) {
      m = i;
      break;
    }
  }
  cout << n - m;
  return 0;
}
