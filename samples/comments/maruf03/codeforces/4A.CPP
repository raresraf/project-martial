#include <stdio.h>

int main() {
  int n, i, j, yes = 0;
  scanf("%d", &n);

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (i % 2 == 0 && j % 2 == 0 && i + j == n) {
        printf("YES\n");
        yes = 1;
        break;
      }
    }
    if (yes == 1) {
      break;
    }
  }
  if (yes == 0) {
    printf("NO\n");
  }

  return 0;
}
