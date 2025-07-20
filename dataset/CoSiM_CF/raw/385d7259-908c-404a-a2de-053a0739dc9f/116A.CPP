#include <stdio.h>
#include <string.h>
int main() {
  int i, n, a, b, t1 = 0, temp = 0, in = 0;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%d %d", &a, &b);
    in = b - a + in;
    if (i == 0 || in > temp) {
      temp = in;
    }
  }
  printf("%d", temp);

  return 0;
}