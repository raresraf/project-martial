#include <ctype.h>
#include <stdio.h>
#include <string.h>
int main() {
  int i, j = 0, k, l, n;
  char ch[125], ch1[250];
  scanf("%s", &ch);
  for (k = 0; k < 125; k++) {
    ch[k] = tolower(ch[k]);
  }
  for (i = 0; i <= 125; i++) {

    if (ch[i] == 'a' || ch[i] == 'e' || ch[i] == 'i' || ch[i] == 'o' ||
        ch[i] == 'u' || ch[i] == 'y')

    {
      l = 0;
    } else {
      l = 1;
    }
    if (l != 0) {
      ch1[j] = '.';
      j++;
      ch1[j] = ch[i];
      j++;
    }
  }
  ch1[strlen(ch1) - 1] = ' ';

  printf("%s", ch1);

  return 0;
}