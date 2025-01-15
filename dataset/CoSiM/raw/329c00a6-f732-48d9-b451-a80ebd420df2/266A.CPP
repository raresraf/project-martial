#include <stdio.h>

int main(void) {

  int n;
  int cnt = 0;
  char prev = 'A';
  char now = 'A';

  scanf("%d", &n);

  for (int i = 0; i <= n; i++) {
    prev = now;
    scanf("%c", &now);

    if (prev == now) {
      cnt++;
    }
  }

  printf("%d", cnt);

  return 0;
}
