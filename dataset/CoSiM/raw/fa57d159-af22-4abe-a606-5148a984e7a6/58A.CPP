#include <stdio.h>
#include <string.h>
int main() {
  char a[100], b[4] = {'0', '0', '0', '0'};
  int i, j = 0, k;
  scanf("%s", a);
  for (i = 0; i < strlen(a); i++)
    if (a[i] == 'h') {
      b[0] = a[i];
      break;
    }
  for (; i < strlen(a); i++)
    if (a[i] == 'e') {
      b[1] = a[i];
      break;
    }
  for (; i < strlen(a); i++) {
    if (a[i] == 'l') {
      b[2] = a[i];
      j++;
    }
    if (j > 1)
      break;
  }
  for (; i < strlen(a); i++) {
    if (a[i] == 'o') {
      b[3] = a[i];
      break;
    }
  }
  if (b[0] == 'h' && b[1] == 'e' && b[2] == 'l' && b[3] == 'o' && j > 1)
    printf("YES");

  else
    printf("NO");
  return 0;
}