#include <stdio.h>
#include <string.h>
int main() {
  int n, i;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    char a[100];
    scanf("%s", a);
    int x = strlen(a);
    if (x <= 10)
      printf("%s\n");
    else
      printf("%c%d%c\n", a[0], x - 2, a[x - 1]);
  }
  return 0;
}