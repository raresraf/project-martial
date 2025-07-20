#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  scanf(" %d", &n);
  int a, b;
  scanf(" %d %d", &a, &b);
  int g = __gcd(a, b);
  if (n % a == 0)
    printf("YES\n%d 0\n", n / a);
  else if (n % b == 0)
    printf("YES\n0 %d\n", n / b);
  else if (n % g == 0) {
    for (int i = 1; i <= (n / a); ++i) {
      int va = n - i * a;
      if (va % b == 0) {
        printf("YES\n%d %d\n", i, va / b);
        return 0;
      }
      if (va < b) {
        puts("NO");
        return 0;
      }
    }
    puts("NO");
  } else
    puts("NO");
  return 0;
}
