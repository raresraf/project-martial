#include <stdio.h>
#include <string.h>
int main() {
  int x = 0, i, n;
  scanf("%d", &n);
  char ch[10];
  for (i = 0; i < n; i++) {
    scanf("%s", ch);
    if (ch[1] == '+' && ch[2] == '+')
      x++;
    else if (ch[1] == '-' && ch[2] == '-')
      x--;
    else if (ch[0] == '-' && ch[1] == '-')
      --x;
    else
      ++x;
  }
  printf("%d", x);

  return 0;
}