#include <bits/stdc++.h>

int main() {
  long long n, i, sum = 0, ans = 0, cnt = 0;
  std::cin >> n;
  long long a[n];

  for (i = 0; i < n; i++) {
    std::cin >> a[i];
    sum += a[i];
  }

  sum /= 2;
  std::sort(a, a + n);

  for (i = n - 1; i >= 0; i--) {
    ans += a[i];
    cnt++;
    if (ans > sum) {
      break;
    }
  }

  std::cout << cnt;

  return 0;
}