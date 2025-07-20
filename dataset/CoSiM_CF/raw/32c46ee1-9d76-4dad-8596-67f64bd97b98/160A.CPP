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
  sort(a, a + n, greater<int>());
  int ans = 0;
  int i;
  for (i = 0; i < n; i++) {
    ans += a[i];
    if (ans > sum / 2)
      break;
  }
  cout << i + 1;
}
