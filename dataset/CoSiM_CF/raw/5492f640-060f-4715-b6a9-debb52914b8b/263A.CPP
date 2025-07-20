/*
  Contest 263
  Problem A : Beautiful Matrix
  Author : Rakesh Kumar
  Date: 18/01/2020
*/

#include <bits/stdc++.h>

int main() {
  int result = 0;
  for (int r = 0; r < 5; ++r) {
    for (int c = 0; c < 5; ++c) {
      int e = 0;
      scanf("%d", &e);

      if (e == 1)
        result = std::abs(2 - r) + std::abs(2 - c);
    }
  }

  printf("%d\n", result);

  return 0;
}
