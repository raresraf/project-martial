#include <stdio.h>
int main() {
  int n, m;
  while (scanf("%d%d", &n, &m) != EOF) {
    printf("%d", n * m / 2);
  }
  return 0;
}
