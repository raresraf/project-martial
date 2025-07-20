#include <stdio.h>
#include <string.h>
int main() {
  char s[101];
  int i, c1 = 0, c2 = 0, c3 = 0;
  scanf("%s", s);
  i = 0;
  while (i < strlen(s)) {
    if (s[i] == '1')
      c1++;
    else if (s[i] == '2')
      c2++;
    else
      c3++;
    i = i + 2;
  }
  for (i = 0; i < c1; i++) {
    if (i == c1 - 1 && c2 == 0 && c3 == 0)
      printf("1");
    else
      printf("1+");
  }
  c1 = 0;
  for (i = 0; i < c2; i++) {
    if (i == c2 - 1 && c1 == 0 && c3 == 0)
      printf("2");
    else
      printf("2+");
  }
  c2 = 0;
  for (i = 0; i < c3; i++) {
    if (i == c3 - 1 && c1 == 0 && c2 == 0)
      printf("3");
    else
      printf("3+");
  }
  return 0;
}