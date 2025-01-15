#include <stdio.h>
#include <string.h>
int main() {
  int i, count = 0;
  char ch[200];
  scanf("%s", ch);
  for (i = 0; i < strlen(ch); i++) {
    if (ch[i] == 'h' && count == 0)
      count++;
    else if (ch[i] == 'e' && count == 1)
      count++;
    else if (ch[i] == 'l' && count == 2)
      count++;
    else if (ch[i] == 'l' && count == 3)
      count++;
    else if (ch[i] == 'o' && count == 4)
      count++;
  }
  if (count == 5)
    printf("YES");
  else
    printf("NO");
  return 0;
}