#include <stdio.h>
#include <string.h>

int main() {
  char string[101];
  int stat = 1;
  int printed = 0;
  scanf("%s", string);
  for (int i = 1; i < strlen(string); i++) {
    if (string[i] == string[i - 1]) {
      stat++;
      if (stat == 7) {
        printf("YES\n");
        printed = 1;
        break;
      }
    } else {
      stat = 1;
    }
  }
  if (printed == 0)
    printf("NO\n");
  return 0;
}
