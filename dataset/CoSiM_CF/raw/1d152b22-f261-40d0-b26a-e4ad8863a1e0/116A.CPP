#include <cstdio>

using namespace std;

int main() {
  int n;
  scanf("%d", &n);
  int i, a, b, tram = 0, max = 0;
  for (i = 0; i < n; i++) {
    scanf("%d %d", &a, &b);
    b = b - a;
    tram += b;
    if (tram > max) {
      max = tram;
    }
  }
  printf("%d", max);
  return 0;
}