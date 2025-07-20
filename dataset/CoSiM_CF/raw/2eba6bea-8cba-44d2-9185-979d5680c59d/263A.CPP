#include <cmath>
#include <cstdio>

int x[5][5], a, b;

using namespace std;

int main() {
  int i, j;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      scanf("%d", &x[i][j]);
      if (x[i][j] == 1) {
        a = i;
        b = j;
        break;
      }
    }
  }
  a = abs(a - 2);
  b = abs(b - 2);
  printf("%d", a + b);
  return 0;
}
