#include <stdio.h>
int main() {
  int n;
  scanf("%d", &n);

  int ara[n][3], count, haha = 0;

  for (int i = 0; i < n; i++) {
    count = 0;
    for (int j = 0; j < 3; j++) {
      scanf("%d", &ara[i][j]);
      if (ara[i][j] == 1) {
        count++;
      }
    }
    if (count >= 2) {
      haha++;
    }
  }

  printf("%d\n", haha);
  return 0;
}
