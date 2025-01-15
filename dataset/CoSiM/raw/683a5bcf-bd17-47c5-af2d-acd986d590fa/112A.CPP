#include <ctype.h>
#include <stdio.h>
#include <string.h>
int main() {
  int i, j, k, l, m;
  char ch1[100], ch2[100];
  scanf("%s\n", ch1);
  scanf("%s", ch2);
  for (j = 0; j < strlen(ch1); j++) {
    ch1[j] = tolower(ch1[j]);
  }
  for (k = 0; k < strlen(ch2); k++) {
    ch2[k] = tolower(ch2[k]);
  }
  for (i = 0; i < strlen(ch1); i++) {
    l = ch1[i];
    m = ch2[i];
    if (l < m) {
      printf("%d", -1);
      break;
    } else if (l > m) {
      printf("%d", 1);
      break;

    } else if (i == strlen(ch1) - 1)
      printf("%d", 0);
  }
  return 0;
}