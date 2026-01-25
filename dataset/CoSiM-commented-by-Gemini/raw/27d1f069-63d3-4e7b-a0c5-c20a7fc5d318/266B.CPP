#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int n, t;

  scanf("%d %d", &n, &t);

  char s[n];

  scanf("%s", s);

  while (t--) {
    for (int i = 0; i < (n - 1); i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        char c = s[i];
        s[i] = s[i + 1];
        s[i + 1] = c;

        i++;
      }
    }
  }

  printf("%s\n", s);

  return 0;
}
