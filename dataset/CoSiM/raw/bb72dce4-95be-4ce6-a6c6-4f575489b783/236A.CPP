#include <stdio.h>
#include <string.h>
int main() {
  char a[100];
  int i, k = 0, j;
  scanf("%s", a);
  for (i = 0; i < strlen(a); i++)
    for (j = i + 1; j < strlen(a); j++)
      if (a[i] == a[j]) {
        k++;
        break;
      }
  i = strlen(a) - k;
  if (i % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
  return 0;
}