/*
  Contest 231
  Problem A : Team
  Author : Rakesh Kumar
  Date: 17/01/2020
*/

#include <bits/stdc++.h>

int main() {
  int n = 0;
  scanf("%d", &n);

  int count = 0;
  while (n--) {
    int a = 0;
    int b = 0;
    int c = 0;
    scanf("%d %d %d", &a, &b, &c);
    if (a + b + c >= 2)
      ++count;
  }

  printf("%d\n", count);

  return 0;
}
