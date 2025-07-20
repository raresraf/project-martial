#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, i, a[100], sum = 0, ans = 0, cnt = 0;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%d", &a[i]);
  }
  sort(a, a + n);
  for (i = 0; i < n; i++) {
    sum += a[i];
  }
  sum = sum / 2;
  while (ans <= sum) {
    ++cnt;
    ans += a[n - cnt];
  }
  printf("%d\n", cnt);
  return 0;
}
