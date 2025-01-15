#include <stdio.h>
int main() {
  int t, a[100], c[100], b[100], i, sa = 0, sb = 0, sc = 0;
  scanf("%d", &t);
  for (i = 0; i < t; i++) {
    scanf("%d%d%d", &a[i], &b[i], &c[i]);

    sa = sa + a[i];
    sb = sb + b[i];
    sc = sc + c[i];
  }
  if (sa == 0 && sb == 0 && sc == 0)

  {
    printf("YES\n");
  } else {
    printf("NO\n");
  }
  return 0;
}
