#include <stdio.h>
int main() {
  int n, i, sum = 0, a, b, c;
  scanf("%d", &n);

  for (i = 1; i <= n; i++) {
    scanf("%d %d %d", &a, &b, &c);
    if ((a == 1 && b == 1) || (b == 1 && c == 1) || (a == 1 && c == 1))
      sum += 1;
  }

  printf("%d\n", sum);
}

/*
Examples
input
3
1 1 0
1 1 1
1 0 0
output
2
input
2
1 0 0
0 1 1
output
1
*/
