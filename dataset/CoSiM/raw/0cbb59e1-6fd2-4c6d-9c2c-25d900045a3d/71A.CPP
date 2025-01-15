#include <stdio.h>
#include <string.h>
int main() {
  int n, i, p;
  scanf("%d", &n);
  char s[n][101];
  for (i = 0; i < n; i++)
    scanf("%s", s[i]);

  printf("\n");
  for (i = 0; i < n; i++) {
    p = strlen(s[i]);
    if (strlen(s[i]) > 10) {
      printf("%c%d%c\n", s[i][0], p - 2, s[i][p - 1]);
    } else
      printf("%s\n", s[i]);
  }
  return 0;
}
