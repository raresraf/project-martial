#include <stdio.h>

int main() {
  int a[100];
  int n;
  int j, p = 0;

  scanf("%d%d", &n, &j);

  for (int i = 0; i < n; i++) {
    scanf("%d", &a[i]);
  }

  for (int i = 0; i < n; i++) {
    if (a[i] >= a[j - 1] && a[i] > 0) {
      p++;
    }
  }

  printf("%d\n", p);

  return 0;
}
