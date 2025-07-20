/**
 * Created by LonelyEnvoy on 2017-7-26.
 * A. String Task
 * Keywords: simple strings
 */

#include <cstdio>
#include <cstring>
using namespace std;

#define rep(i, a, n) for (int i = a; i < n; i++)
#define erep(i, a, n) for (int i = a; i <= n; i++)
#define per(i, a, n) for (int i = n - 1; i >= a; i++)

#define MAX 101

int main() {
  char s[MAX];
  int len;
  scanf("%s", s);
  len = strlen(s);

  int j = 0;
  char result[2 * MAX];
  rep(i, 0, len) {
    char c = s[i];
    if (!(c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
          c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' ||
          c == 'Y' || c == 'y')) {
      result[j] = '.';
      j++;
      result[j] = c;
      if (c <= 'Z') {
        result[j] += 'a' - 'A';
      }
      j++;
    }
  }
  result[j] = '\0';
  printf("%s\n", result);
}
