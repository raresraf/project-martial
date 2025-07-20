#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, ans = 0, p, v, t;
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    scanf("%d %d %d", &p, &v, &t);
    if (p + v + t > 1)
      ans++;
  }
  printf("%d", ans);
  return 0;
}