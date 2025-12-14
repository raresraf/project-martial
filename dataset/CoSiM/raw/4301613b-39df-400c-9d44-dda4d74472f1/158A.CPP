#include <stdio.h>

int main() {
  int n, k;

  scanf("%d%d", &n, &k);

  int a;

  for (int i = 0; i < k; i++) {
    scanf("%d", &a);
    if (a == 0) {
      printf("%d", i);
      return 0;
    }
  }
  int b;
  for (int i = k; i < n; i++) {
    scanf("%d", &b);
    if (b != a) {
      printf("%d", i);
      return 0;
    }
  }

  printf("%d", n);
  return 0;
}
