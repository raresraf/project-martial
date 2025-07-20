#include <stdio.h>
#include <string.h>

int main() {
  char ch[105];
  int len, i;

  scanf("%s", ch);

  len = strlen(ch);
  for (i = 0; i < len; i++) {
    if (ch[i] >= 'A' && ch[i] <= 'Z')
      ch[i] = 97 + (ch[i] - 'A');
    if (ch[i] == 'a' || ch[i] == 'e' || ch[i] == 'i' || ch[i] == 'o' ||
        ch[i] == 'u' || ch[i] == 'y')
      continue;
    else
      printf(".%c", ch[i]);
  }
  printf("\n");

  return 0;
}
