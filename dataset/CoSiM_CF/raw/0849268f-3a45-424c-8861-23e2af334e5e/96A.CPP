#include <stdio.h>
#include <string.h>
int main() {
  int i, j;
  char a[100];
  scanf("%s", a);
  for (i = 0; i < strlen(a); i++) {
    for (j = i; j <= i + 6; j++) {
      if (a[j] == a[j + 1])
        continue;
      else
        break;
    }

    if (j == i + 6) {
      printf("YES");
      break;
    }
  }

  if (i == strlen(a))
    printf("NO");
  return 0;
}