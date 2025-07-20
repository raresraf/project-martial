#include <bits/stdc++.h>
/*
0 0 0 0 0
0 0 0 0 1
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
*/

int main() {
  int val;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      std::cin >> val;
      if (val == 1) {
        std::cout << abs(i - 3) + abs(j - 3);
      }
    }
  }
  return 0;
}
