#include <stdio.h>

int main(void) {

  int n; //
  int k; //       ?

  int a[50];

  scanf("%d %d", &n, &k);
  for (int i = 0; i < n; i++) {
    scanf("%d ", &a[i]);
  }

  if (a[k - 1] == 0) { //              0    , 0                 .
    if (a[0] == 0) {
      printf("%d", a[0]);
      return 0;
    }
    for (int j = k - 1; j >= 0; j--) {
      if (a[j] != 0) {
        printf("%d", j + 1);
        return 0;
      }
    }
  } else { //     0       ,                .
    for (int j = k - 1; j < n; j++) {
      if (a[j] > a[j + 1]) {
        printf("%d", j + 1);
        return 0;
      }
    }
    printf("%d", n);
  }
  return 0;
}
