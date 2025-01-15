#include <stdio.h>

int main() {
  int k, n, w, i, cost = 0, loan, rate;
  scanf("%d %d %d", &k, &n, &w);
  for (i = 1; i <= w; i++) {
    rate = i * k;
    cost = cost + rate;
  }
  loan = cost - n;
  if (loan <= 0)
    loan = 0;
  printf("%d", loan);
}
