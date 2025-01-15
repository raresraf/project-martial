#include <stdio.h>

int main() {
  int n, k, max = 0, i, count = 0, ara[100];
  scanf("%d %d", &n, &k);
  for (i = 0; i < n; i++) {
    scanf("%d", &ara[i]);
  }
  for (i = 0; i < n; i++) {
    if (i + 1 == k) {
      max = ara[i];
    }
  }
  for (i = 0; i < n; i++) {
    if (ara[i] >= max && ara[i] > 0) {
      count++;
    }
  }
  printf("%d\n", count);
}
