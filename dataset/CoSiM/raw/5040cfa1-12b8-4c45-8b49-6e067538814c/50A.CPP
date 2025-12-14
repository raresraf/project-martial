#include <stdio.h>

int main(void) {

  int m;
  int n;

  scanf("%d %d", &m, &n);

  if (m * n % 2 == 0) {
    printf("%d", m * n / 2);
  } else {
    printf("%d", (m * n - 1) / 2);
  }

  return 0;
}
