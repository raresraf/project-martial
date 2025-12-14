#include <stdio.h>
using namespace std;

int main() {
  int n, a, b, temp = 0, max = 0;
  scanf("%d", &n);
  while (n--) {
    scanf("%d %d", &a, &b);
    temp += b - a;
    if (temp > max)
      max = temp;
  }
  printf("%d", max);
  return 0;
}
