#include <stdio.h>
#include <string.h>
#define SIZE 100
int main() {
  char a[SIZE], b[SIZE];
  int len, i, j = 0;
  scanf("%s", a);
  len = strlen(a);

  for (i = 0; i < len; i++) {
    if (a[i] >= 65 && a[i] <= 90) {
      a[i] = a[i] + 32;
    }
  }
  for (i = 0; i < len; i++) {
    if (a[i] == 'a' || a[i] == 'e' || a[i] == 'i' || a[i] == 'o' ||
        a[i] == 'u' || a[i] == 'y') {
      continue;
    } else {
      b[j] = '.';
      b[j + 1] = a[i];
      j = j + 2;
    }
  }

  printf("%s", b);
}
