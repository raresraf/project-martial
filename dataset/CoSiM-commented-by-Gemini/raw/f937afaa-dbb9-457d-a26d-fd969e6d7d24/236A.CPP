#include <stdio.h>
#include <string.h>
int main() {
  int i, j, small, count = 0;
  char temp, temp1, ch[200], ch1[200];
  scanf("%s", ch);
  for (i = 0; i < strlen(ch); i++) {
    for (j = i + 1; j < strlen(ch); j++) {
      if (ch[j] > ch[i]) {
        temp = ch[i];
        ch[i] = ch[j];
        ch[j] = temp;
      }
    }
  }
  temp1 = ch[0];
  for (i = 0; i < strlen(ch); i++) {
    if (ch[i] != temp) {
      count++;
      temp = ch[i];
    }
  }
  if (count % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
  return 0;
}