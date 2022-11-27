#include <stdio.h>

int main(void) {

  int w;

  scanf("%d", &w);

  for (int i = 1; i < w; i++) {
    if (i % 2 == 0 && (w - i) % 2 == 0) {
      printf("YES");
      return 0;
    }
  }
  printf("NO");
  return 0;
}
