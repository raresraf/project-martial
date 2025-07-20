#include <stdio.h>
int main() {
  int n, k;
  scanf("%d %d", &n, &k);
  int ara[n];

  for (int i = 0; i < n; i++) {
    scanf("%d", &ara[i]);
  }

  int count = 0;

  for (int i = 0; i < n; i++) {
    if (ara[i] > 0) {
      if (ara[i] >= ara[k - 1])
        count++;
    }
  }
  printf("%d\n", count);
  return 0;
}
