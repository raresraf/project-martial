#include <stdio.h>
#include <string.h>

int main() {
  char ara[10000];
  scanf("%s", ara);

  if (ara[0] >= 'a' && ara[0] <= 'z')
    ara[0] = ara[0] - 32;

  printf("%s\n", ara);
  return 0;
}
