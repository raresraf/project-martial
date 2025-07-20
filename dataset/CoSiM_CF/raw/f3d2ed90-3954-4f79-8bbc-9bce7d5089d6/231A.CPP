#include <stdio.h>
int main() {
  int n, k = 0;
  scanf("%d", &n);
  while (n--) {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    int s = a + b + c;
    if (s >= 2)
      k++;
  }
  printf("%d", k);
  return 0;
}
