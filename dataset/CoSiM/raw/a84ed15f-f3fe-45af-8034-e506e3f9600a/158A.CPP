#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  int n, k, i, num, cnt = 0, a;

  scanf("%d %d", &n, &k);

  for (i = 1; i <= n; i++) {
    scanf("%d", &num);
    if (i < k && num > 0)
      cnt++;
    else if (i == k) {
      a = num;
      if (num > 0)
        cnt++;
    } else if (i > k && a > 0)
      if (num == a)
        cnt++;
  }

  printf("%d\n", cnt);

  return 0;
}
