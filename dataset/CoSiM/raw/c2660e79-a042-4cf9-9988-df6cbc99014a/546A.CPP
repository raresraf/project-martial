#include <stdio.h>

using namespace std;
int main() {
  int k, n, w, i, sum = 0;
  scanf("%d %d %d", &k, &n, &w);
  for (i = 1; i <= w; i++) {
    sum = sum + (k * i);
  }
  if (n < sum) {
    printf("%d\n", sum - n);
  } else {
    printf("%d\n", 0);
  }
}
