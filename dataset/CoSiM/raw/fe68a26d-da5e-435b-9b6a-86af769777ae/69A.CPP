/* 2013
 * Maciej Szeptuch
 * II UWr
 */
#include <cstdio>

int n, m;

bool check(int n, int m);
bool isPrime(int n);

int main(void) {
  scanf("%d %d", &n, &m);
  puts(check(n + 1, m) ? "YES" : "NO");
  return 0;
}

bool check(int n, int m) {
  if (!isPrime(m))
    return false;

  while (n < m)
    if (isPrime(n++))
      return false;

  return true;
}

bool isPrime(int n) {
  for (int k = 2; k < n; ++k)
    if (n % k == 0)
      return false;

  return true;
}
