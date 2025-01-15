#include <stdio.h>
int main() {
  char str[1000];
  scanf("%s", str);
  int count = 0;
  for (int i = 0; str[i] != '\0'; i++) {
    if (str[i] == '4' || str[i] == '7')
      count++;
  }
  (count == 4 || count == 7) ? printf("YES\n") : printf("NO\n");
}
