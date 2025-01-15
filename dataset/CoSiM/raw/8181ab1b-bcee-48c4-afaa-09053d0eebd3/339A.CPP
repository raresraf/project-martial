#include <stdio.h>
#include <string.h>
int main() {
  int i, j;
  char temp, ch[100];
  scanf("%s", ch);
  for (i = 0; i < strlen(ch); i++) {
    for (j = i + 1; j < strlen(ch); j++) {
      if (ch[j] < ch[i] && ch[j] != '+') {
        temp = ch[j];
        ch[j] = ch[i];
        ch[i] = temp;
      }
    }
  }
  printf("%s", ch);
  return 0;
}