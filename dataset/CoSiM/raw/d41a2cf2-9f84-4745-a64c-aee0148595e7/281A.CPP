#include <ctype.h>
#include <stdio.h>
#include <string.h>
int main() {
  char ch[1000];
  scanf("%s", ch);
  ch[0] = toupper(ch[0]);
  printf("%s", ch);
  return 0;
}