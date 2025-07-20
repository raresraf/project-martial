#include <cstdio>

using namespace std;

int main() {
  int x, y, z, count = 0, n;
  scanf("%d", &n);
  while (n--) {
    scanf("%d %d %d", &x, &y, &z);
    if ((x + y + z) >= 2) {
      count++;
    }
  }
  printf("%d\n", count);
  return 0;
}
