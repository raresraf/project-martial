#include <stdio.h>

int main() {
  int n, total = 0, out = 0, in = 0, max = 0;
  scanf("%d", &n);
  while (n--) {
    scanf("%d %d", &out, &in);
    total -= out;
    total += in;
    if (total > max) {
      max = total;
    }
  }
  printf("%d\n", max);
}
