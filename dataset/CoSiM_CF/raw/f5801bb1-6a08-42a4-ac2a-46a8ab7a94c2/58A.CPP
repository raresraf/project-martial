/* 2011
 * Maciej Szeptuch
 * XIV LO Wroc aw
 */
#include <algorithm>
#include <cstdio>
//#define DEBUG(args...) fprintf(stderr, args)
#define DEBUG(args...)

int mod[4], start, end, result;

inline static bool check(const int &a);

int main(void) {
  scanf("%d %d %d %d %d %d", &mod[0], &mod[1], &mod[2], &mod[3], &start, &end);
  std::sort(mod, mod + 4);
  for (int x = start; x <= end; ++x)
    result += check(x);

  printf("%d\n", result);
  return 0;
}

inline static bool check(const int &a) {
  int count = 0;
  while (std::next_permutation(mod, mod + 4) && count < 7)
    count += a == (((a % mod[0]) % mod[1]) % mod[2]) % mod[3];

  return count == 7;
}
