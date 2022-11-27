/* 2011
 * Maciej Szeptuch
 * XIV LO Wroc aw
 */
#include <cstdio>
//#define DEBUG(args...) fprintf(stderr, args)
#define DEBUG(args...)

int number;

int main(void) {
  scanf("%d", &number);
  puts((number > 2 && !(number & 1)) ? "YES" : "NO");
  return 0;
}
