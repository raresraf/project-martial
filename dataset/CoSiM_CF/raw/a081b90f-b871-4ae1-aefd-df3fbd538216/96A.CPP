#include <stdio.h>
#include <string.h>

int main() {
  char ch[105];
  int len, i, flag = 0, cnt1 = 0, cnt2 = 0;
  scanf("%s", &ch);
  len = strlen(ch);
  for (i = 0; i < len; i++) {
    if (ch[i] == '1') {
      cnt1++;
      cnt2 = 0;
      if (cnt1 == 7) {
        flag = 1;
        break;
      }
    } else if (ch[i] == '0') {
      cnt1 = 0;
      cnt2++;
      if (cnt2 == 7) {
        flag = 1;
        break;
      }
    }
  }
  if (flag)
    printf("YES\n");
  else
    printf("NO\n");

  return 0;
}
