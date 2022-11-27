#include <cstdio>

using namespace std;

int main() {
  long long x;
  scanf("%I64d", &x);
  if (x == 2) {
    printf("NO");
    return 0;
  }
  if (x % 2 == 0) {
    printf("YES");
  } else
    printf("NO");
  return 0;
}
