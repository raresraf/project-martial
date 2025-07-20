#include <stdio.h>
int main() {
  int n;
  scanf("%d", &n);

  int ara[n][3];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      scanf("%d", &ara[i][j]);
    }
  }
  int sum = 0, prev, flag = 1;
  for (int j = 0; j < 3; j++) {
    prev = sum;
    sum = 0;
    for (int i = 0; i < n; i++) {
      sum = sum + ara[i][j];
    }
    if (prev == sum)
      continue;
    else {
      flag = 0;
      break;
    }
  }
  if (flag == 1)
    printf("YES\n");
  if (flag == 0)
    printf("NO\n");

  return 0;
}
