#include <cstdio>

int main() {
  int k, n, w;
  scanf("%d %d %d", &k, &n, &w);

  int money = w * (w + 1) / 2 * k;

  printf("%d\n", money <= n ? 0 : money - n);
  return 0;
}
