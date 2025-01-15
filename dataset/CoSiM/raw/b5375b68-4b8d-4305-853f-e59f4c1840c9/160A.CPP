/// Problem Name: Twins
/// Problem Link: https://codeforces.com/problemset/problem/160/A

#include <bits/stdc++.h>

using namespace std;

int A, B;
int a[101];
int b[101];

int main() {
  int n, sum = 0;
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    scanf("%d", &a[i]);
    sum += a[i];
  }
  sort(a, a + n);
  int ans = 0;
  int S = 0;
  for (int i = n - 1; i >= 0; i--) {
    sum -= a[i];
    S += a[i];
    ans++;
    if (sum < S)
      break;
  }
  printf("%d\n", ans);
  return 0;
}
