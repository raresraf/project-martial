#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  scanf("%d", &n);
  while (n--) {
    long long int a, b, x, temp;
    scanf("%lld %lld", &a, &b);
    if (a == 0 || b == 0 || (a == 1 && b == 1)) {
      printf("0\n");
    } else if ((a == 1 && b > 1) || (b == 1 && a > 1)) {
      printf("1\n");
    } else {
      if (b > a) {
        temp = a;
        a = b;
        b = temp;
      }
      x = (a + b) / 3;
      if (x > b) {
        printf("%lld\n", b);
      } else {
        printf("%lld\n", x);
      }
    }
  }
  return 0;
}
