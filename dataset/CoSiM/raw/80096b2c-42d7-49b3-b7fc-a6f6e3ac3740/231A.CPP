#include <cstdio>
using namespace std;

int main() {
  int n, a, b, c, x = 0;
  scanf("%d", &n);
  while (n--) {
    a = 0;
    b = 0;
    c = 0;
    scanf("%d%d%d", &a, &b, &c);
    if (a + b + c >= 2)
      x++;
  }
  printf("%d", x);
}
