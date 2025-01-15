#include <stdio.h>
#include <string.h>

int main() {
  int n, l;
  char s[100];
  scanf("%d", &n);
  while (n--) {
    scanf("%s", &s);
    l = strlen(s);
    if (l <= 10)
      printf("%s\n", s);
    else
      printf("%c%d%c\n", s[0], l - 2, s[l - 1]);
  }
  return 0;
}
