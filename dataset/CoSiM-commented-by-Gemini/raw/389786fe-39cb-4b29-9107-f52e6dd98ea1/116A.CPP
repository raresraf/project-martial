#include <cstdio>
using namespace std;

int main() {
  int n, a, b, x = 0, capacity = 0;
  scanf("%d", &n);
  while (n--) {
    scanf("%d%d", &a, &b);
    x = x - a + b;
    if (x > capacity)
      capacity = x;
  }
  printf("%d", capacity);
}
