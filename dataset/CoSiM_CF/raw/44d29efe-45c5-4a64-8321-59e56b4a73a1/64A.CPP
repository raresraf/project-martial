#include <stdio.h>
#include <string.h>

int main() {
  int t, sum = 0, c;
  char m[20];
  scanf("%d", &t);
  while (t--) {
    scanf("%s", m);
    if (!strcmp(m, "donate")) {
      scanf("%d", &c);
      sum += c;
    } else {
      printf("%d\n", sum);
    }
  }
  return 0;
}
