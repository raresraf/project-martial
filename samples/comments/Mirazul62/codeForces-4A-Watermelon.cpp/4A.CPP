#include <stdio.h>
int main() {
  int w;

  scanf("%d", &w);
  if (w <= 2)
    printf("NO");
  if (w > 2)

  {

    if (w % 2 == 0) {
      printf("YES");
    } else {
      printf("NO");
    }
  }
}
