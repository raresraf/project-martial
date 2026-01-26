#include <cctype>
#include <cstdio>

using namespace std;

int main() {
  char c;
  int count = 0;
  while (1) {
    fscanf(stdin, "%c", &c);
    if (!isdigit(c)) {
      break;
    }
    if (c - '0' == 4 || c - '0' == 7) {
      ++count;
    }
  }
  int m = 10, n = 1;
  int dig;

  if (count == 0) {
    fprintf(stdout, "NO");
    return 0;
  }

  while (count / n > 0) {
    dig = (count % m) / n;
    if (dig != 4 && dig != 7) {
      fprintf(stdout, "NO");
      return 0;
    }
    m = m * 10;
    n = n * 10;
  }
  fprintf(stdout, "YES");
}
