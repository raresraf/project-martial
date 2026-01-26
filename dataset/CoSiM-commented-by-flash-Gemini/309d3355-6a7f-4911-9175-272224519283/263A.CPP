#include <cstdio>
#include <cstdlib>

using namespace std;

int main() {
  char ch;
  int r = 1, c = 1;

  while ((ch = getchar()) != '1') {
    if (ch == '0')
      c++;

    /* Better not try converting 1D to 2D */
    if (c == 6) {
      c = 1;
      r++;
    }
  }

  c = abs(3 - c) + abs(3 - r);

  printf("%d\n", c);
  return 0;
}
