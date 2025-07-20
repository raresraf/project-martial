#include <stdio.h>
#include <stdlib.h>

int main() {
  int i, a, b, n;
  for (i = 0; i < 25; i++) {
    scanf("%d", &n);
    if (n) {
      a = i / 5;
      b = i % 5;
    };
  }
  printf("%d", abs(2 - a) + abs(2 - b));
  return 0;
}