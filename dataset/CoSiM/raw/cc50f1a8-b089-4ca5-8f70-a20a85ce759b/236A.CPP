#include <cstdio>

using namespace std;

int main() {
  char s[101];
  fscanf(stdin, "%s", s);
  int i, j, count = 0;

  for (i = 0; s[i] != '\0'; ++i) {
    if (s[i] == 1) {
      --count;
    }
    for (j = i + 1; s[j] != '\0'; ++j) {
      if (s[j] == s[i]) {
        s[j] = 1;
      }
    }
    ++count;
  }
  if (count % 2 == 0) {
    fprintf(stdout, "CHAT WITH HER!");
  } else {
    fprintf(stdout, "IGNORE HIM!");
  }

  return 0;
}
