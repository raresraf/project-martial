#include <stdio.h>
int main() {
  int n, x = 0;
  scanf("%d", &n);
  while (n--) {
    char b[100] = " ";
    scanf("%s", b);
    if (b[0] == '+' || b[1] == '+')
      x++;
    else
      x--;
  }
  printf("%d", x);
  return 0;
}
