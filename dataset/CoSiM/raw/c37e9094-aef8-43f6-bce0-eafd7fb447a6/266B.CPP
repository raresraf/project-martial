#include <cstdio>

using namespace std;

int main() {
  int n, row, i, j;
  scanf("%d", &n);
  char x[n + 1], temp;
  scanf("%d", &row);
  scanf("%s", x);
  for (i = 0; i < row; i++) {
    for (j = 0; j + 1 < n; j++) {
      if (x[j] < x[j + 1]) {
        temp = x[j];
        x[j] = x[j + 1];
        x[j + 1] = temp;
        j++;
      }
    }
  }
  printf("%s", x);
  return 0;
}
