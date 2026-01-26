#include <stdio.h>
int main() {
  int a, i, m, n, o, sum1 = 0, sum2 = 0, sum3 = 0;
  scanf("%d", &a);
  for (i = 0; i < a; i++) {
    scanf("%d", &m);
    sum1 += m;
    scanf("%d", &n);
    sum2 += n;
    scanf("%d", &o);
    sum3 += o;
  }
  if (sum1 == 0 && sum2 == 0 && sum3 == 0)
    printf("YES\n");
  else
    printf("NO\n");
}