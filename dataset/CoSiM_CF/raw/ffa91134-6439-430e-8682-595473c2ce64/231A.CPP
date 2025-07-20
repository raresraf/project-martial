#include "stdio.h"

int main() {
  int n;
  scanf("%d", &n);

  int p, v, t;
  p = v = t = 0;

  int count = 0;
  for (int i = 0; i < n; i++) {
    scanf("%d%d%d", &p, &v, &t);

    if ((p + v + t) >= 2) {
      count++;
    }
  }

  printf("%d", count);

  return 0;
}