#include <stdio.h>
#include <string.h>
int main() {
  char s[100];
  scanf("%s", s);
  int length = strlen(s), count, i, j, flag;

  for (i = 0; i < length; i++) {
    count = 0, flag = 0;
    if (s[i] == '0') {
      count++;
      for (j = i + 1; j < length; j++) {
        if (s[j] == '0') {
          count++;
          if (count >= 7) {
            flag = 1;
            printf("YES\n");
            return 0;
          }
        } else {
          i = j - 1;
          break;
        }
      }
    }
    if (s[i] == '1') {
      count++;
      for (j = i + 1; j < length; j++) {
        if (s[j] == '1') {
          count++;
          if (count >= 7) {
            flag = 1;
            printf("YES\n");
            return 0;
          }
        } else {
          i = j - 1;
          break;
        }
      }
    }
  }
  if (flag == 0)
    printf("NO\n");
  return 0;
}
