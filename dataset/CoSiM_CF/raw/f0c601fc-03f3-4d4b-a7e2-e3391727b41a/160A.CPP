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

  int count = 0, ans = 0;
  sort(a, a + n);

  for (int i = n - 1; i >= 0; i--) {
    ans += a[i];
    count++;
    if (ans > sum - ans) {
      break;
    }
  }
  cout << count;

  return 0;
}
