#include <stdio.h>
int main() {
  long long n, w, k, s, ans;
  scanf("%lld %lld %lld", &k, &n, &w);
  s = k * (w * (w + 1) / 2);
  ans = s - n;
  if (ans > 0) {
    printf("%lld", ans);
  } else
    printf("0");
  return 0;
}