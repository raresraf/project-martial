#include <bits/stdc++.h>

typedef long long ll;

const int sz = 2e5;
ll seq[sz];

inline long sign(ll data) {
  if (data > 0)
    return 1;
  return -1;
}
int main() {
  int tests;
  scanf("%d", &tests);
  while (tests--) {
    long n, i;
    scanf("%ld", &n);
    for (i = 0; i < n; i++)
      scanf("%lld", seq + i);
    ll sum = 0, maxx;
    for (i = 0; i < n; i++) {
      if (i == 0) {
        maxx = seq[i];
      } else {
        if (sign(seq[i - 1]) + sign(seq[i]) == 0) {
          sum += maxx;
          maxx = seq[i];
        } else {
          maxx = std::max(maxx, seq[i]);
        }
      }
    }
    sum += maxx;
    printf("%lld\n", sum);
  }
  return 0;
}