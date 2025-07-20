#include <stdio.h>
#include <string.h>
int main() {
  int ln, i, j = 0, ck = 0;
  char s[105];
  char name[10] = "hello";
  scanf("%s", s);
  ln = strlen(s);
  for (i = 0; i < ln; i++) {
    if (s[i] == name[j]) {
      j++;
      ck++;
    }
  }
  if (ck == 5)
    printf("YES");
  else
    printf("NO");
}