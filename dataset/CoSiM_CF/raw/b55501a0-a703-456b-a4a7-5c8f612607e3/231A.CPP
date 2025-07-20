#include <cstdio>

using namespace std;

int main() {
  int n, a, b, c, count = 0;
  scanf("%d", &n);
  for (int i = 0; i < n; ++i) {
    scanf("%d%d%d", &a, &b, &c);
    if (a + b + c > 1)
      count++;
  }
  printf("%d\n", count);
}