#include <bits/stdc++.h>
using namespace std;

int main() {
  int matrix[5][5];
  int a;
  int x, y;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      scanf("%d", &a);
      if (a == 1) {
        x = i;
        y = j;
      }
    }
  }

  int ans = abs(x - 2) + abs(y - 2);

  printf("%d\n", ans);

  return 0;
}
