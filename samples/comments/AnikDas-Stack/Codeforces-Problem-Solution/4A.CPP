#include <stdio.h>
int main() {
  int w, flag = 0;
  scanf("%d", &w);

  if (w % 2 != 0) {
    printf("NO\n");
    return 0;
  }

  for (int i = 2; i < w; i = i + 2) {
    if ((w - i) % 2 == 0) {
      flag = 1;
      printf("YES\n");
      break;
    }
  }
  if (flag == 0)
    printf("NO\n");
  return 0;
}
