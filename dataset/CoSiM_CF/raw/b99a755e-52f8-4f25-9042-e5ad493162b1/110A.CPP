#include <stdio.h>

using namespace std;

int main() {
  long long int n;
  int lucky = 0;
  scanf("%I64d", &n);
  while (n > 0) {
    if (n % 10 == 4 || n % 10 == 7)
      lucky++;
    n /= 10;
  }
  if (lucky == 4 || lucky == 7)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}