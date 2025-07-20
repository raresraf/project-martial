#include <stdio.h>
int main() {
  long long int n;
  int i, count = 0, j, k;
  scanf("%lld", &n);
  for (i = 0; n != 0; i++) {
    k = n % 10;
    n = n / 10;
    if (k == 4 || k == 7)
      count++;
  }
  i = count;
  if (i) {
    while (i) {
      k = i % 10;

      if (k != 4 && k != 7) {
        printf("NO");
        break;
      }
      i = i / 10;
    }
    if (i == 0)
      printf("YES");
  } else
    printf("NO");
  return 0;
}