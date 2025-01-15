#include <cstdio>

using namespace std;

int main() {
  char s[101];
  fscanf(stdin, "%s", s);
  int count = 1;

  for (int i = 0; s[i + 1] != '\0'; ++i) {
    if (s[i] == s[i + 1]) {
      ++count;
    } else {
      count = 1;
    }
    if (count == 7) {
      fprintf(stdout, "YES");
      return 0;
    }
  }
  fprintf(stdout, "NO");
  return 0;
}
