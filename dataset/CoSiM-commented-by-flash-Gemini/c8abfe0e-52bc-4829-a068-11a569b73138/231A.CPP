#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  int n, a, b, c, sum, cnt = 0;

  scanf("%d", &n);

  while (n--) {
    sum = 0;
    scanf("%d %d %d", &a, &b, &c);
    sum = a + b + c;
    if (sum > 1)
      cnt++;
  }

  printf("%d\n", cnt);

  return 0;
}
