#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  int m, n, ans;

  scanf("%d %d", &m, &n);

  if (n > m) {
    ans = m;
    m = n;
    n = ans;
  }

  if ((n % 2 == 0)) {
    ans = (n / 2) * m;
  } else if (n % 2 != 0) {
    ans = ((n / 2) * m) + (m / 2);
  }

  printf("%d\n", ans);

  return 0;
}
