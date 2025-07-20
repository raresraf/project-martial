#include <stdio.h>

using namespace std;
int main() {
  int n, a, b;
  scanf("%d", &n);
  int maxim = 0, sum = 0;
  while (n--) {
    scanf("%d %d", &a, &b);
    sum = sum + b - a;
    if (sum > maxim) {
      maxim = sum;
    }
  }
  printf("%d\n", maxim);
}
