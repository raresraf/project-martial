#include <stdio.h>
int main() {
  long long a;
  scanf("%lld", &a);
  printf(a % 2 == 0 && a > 2 ? "YES\n" : "NO\n");
}
