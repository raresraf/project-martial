#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int a[n];
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  sort(a, a + n);
  int s = 0;
  int i;
  for (i = n - 1; i >= 0; i--) {
    s += a[i];
    sum -= a[i];
    if (s > sum)
      break;
  }
  cout << n - i;
}