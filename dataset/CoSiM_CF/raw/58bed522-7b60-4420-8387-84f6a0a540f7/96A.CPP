/* 2012
 * Maciej Szeptuch
 * II UWr
 */
#include <cstdio>

int c;

int main(void) {
  while ((c = getchar()) != -1)
    if (c == 'H' || c == 'Q' || c == '9') {
      puts("YES");
      return 0;
    }

  puts("NO");
  return 0;
}
