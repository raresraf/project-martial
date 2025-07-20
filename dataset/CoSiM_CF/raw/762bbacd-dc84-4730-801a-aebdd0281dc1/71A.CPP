#include <stdio.h>
#include <string.h>

int main() {
  int a, i, p, b;
  char s[105];
  scanf("%d", &a);
  while (a--) {
    scanf("%s", s);
    p = strlen(s);
    if (p <= 10)
      printf("%s\n", s);
    else
      printf("%c%d%c\n", s[0], p - 2, s[p - 1]);
  }
}