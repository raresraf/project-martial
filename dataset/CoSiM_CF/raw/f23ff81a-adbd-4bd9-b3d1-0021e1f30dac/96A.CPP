#include <stdio.h>
#include <string.h>
int main() {
  int i, t, c = 0;
  char a[100];
  scanf("%s", a);
  for (i = 0; i < strlen(a); i++) {
    if (i == 0 || t != a[i]) {
      t = a[i];
      c = 0;
    } else {
      c++;
    }
    if (c == 6) {
      printf("YES");
      break;
    }
  }
  if (c != 6) {
    printf("NO");
  }

  return 0;
}