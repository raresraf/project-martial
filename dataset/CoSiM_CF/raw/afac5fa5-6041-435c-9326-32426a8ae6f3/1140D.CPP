#include <bits/stdc++.h>

int main() {
  int n;
  scanf("%d", &n);
  // split the regular polygon across vertex 1 into triangles
  // as 1-2-3, 1-3-4, 1-4-5 etc
  unsigned long long sum = 0;
  sum += (n - 1) * n * (2 * n - 1) / 6;
  sum += n * (n - 1) / 2;
  sum -= 2;
  printf("%llu\n", sum);
  return 0;
}