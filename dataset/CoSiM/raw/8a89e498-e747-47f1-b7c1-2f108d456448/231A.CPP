/// Problem Name: Team
/// Problem Link: https://codeforces.com/problemset/problem/231/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, ans = 0;
  scanf("%d", &n);
  int a, b, c;
  while (n--) {
    scanf("%d %d %d", &a, &b, &c);
    if ((a && b) || (b && c) || (c && a))
      ans++;
  }
  printf("%d\n", ans);
  return 0;
}
