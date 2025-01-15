#include <stdio.h>
#include <string.h>
int main() {
  char s[105];
  int ck = 0, ln, i;
  scanf("%s", s);
  ln = strlen(s);
  for (i = 0; i < ln; i++) {
    if (s[i] == '0' && s[i + 1] == '0' && s[i + 2] == '0' && s[i + 3] == '0' &&
        s[i + 4] == '0' && s[i + 5] == '0' && s[i + 6] == '0') {
      ck = 1;
      break;
    } else if (s[i] == '1' && s[i + 1] == '1' && s[i + 2] == '1' &&
               s[i + 3] == '1' && s[i + 4] == '1' && s[i + 5] == '1' &&
               s[i + 6] == '1') {
      ck = 1;
      break;
    }
  }
  if (ck == 1)
    printf("YES");
  else
    printf("NO");
}