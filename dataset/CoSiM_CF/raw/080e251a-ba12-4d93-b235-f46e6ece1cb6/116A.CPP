// Codeforces
// 116A - Tram

#include <cstdio>

int main() {
  int numStops = 0;
  scanf("%d", &numStops);

  int total = 0, max = 0, in = 0, out = 0;

  while (numStops--) {
    scanf("%d %d", &out, &in);
    total += in - out;

    if (total > max)
      max = total;
  }

  printf("%d", max);
  return 0;
}
