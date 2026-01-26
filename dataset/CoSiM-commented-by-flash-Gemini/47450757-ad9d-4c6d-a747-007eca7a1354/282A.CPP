#include <stdio.h>
int main() {
  int i, j, a, n = 0;
  char str[20];
  while (scanf("%d", &a) != EOF) {
    for (i = 0; i < a; i++) {
      scanf("%s", str);
      if (str[1] == '+' || str[2] == '+')
        n++;
      else if (str[1] == '-' || str[2] == '-')
        n--;
    }
    printf("%d\n", n);
    n = 0;
  }
  return 0;
}
