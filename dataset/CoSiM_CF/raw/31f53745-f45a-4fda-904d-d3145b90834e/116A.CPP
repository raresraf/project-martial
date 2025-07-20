/*
  Contest 116
  Problem A : Tram
  Author : Rakesh Kumar
  Date: 09/04/2020
*/

#include <bits/stdc++.h>

int main() {
  int n = 0;
  scanf("%d", &n);

  int result = 0;
  int total = 0;
  while (n--) {
    int a = 0;
    int b = 0;
    scanf("%d %d", &a, &b);
    total = (total - a) + b;
    result = std::max(result, total);
  }

  printf("%d\n", result);

  return 0;
}
