#include <bits/stdc++.h>

using namespace std;

int main() {
  int m[5][5], r, c;
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++)
      scanf("%d", &m[i][j]);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (m[i][j] == 1) {
        r = i;
        c = j;
        break;
      }
    }
  }
  printf("%d\n", abs(2 - r) + abs(2 - c));
  return 0;
}