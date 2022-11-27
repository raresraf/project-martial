#include <stdio.h>
int main() {
  int i, j;
  scanf("%d", &i);
  if (i % 2 == 0 && i > 2) {
    printf("yes");
  } else {
    printf("no");
  }
  return 0;
}
