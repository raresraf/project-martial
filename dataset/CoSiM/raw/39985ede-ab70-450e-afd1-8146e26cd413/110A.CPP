#include <stdio.h>
int main() {
  long long int n, rem;
  scanf("%lld", &n);
  int count = 0;

  while (n != 0) {
    rem = n % 10;
    if (rem == 7 || rem == 4) {
      count++;
    }
    n = n / 10;
  }
  if (count == 4 || count == 7)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
