#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;

char s[105];
int main(int argc, char **argv) {
  scanf("%s", s);
  int len = strlen(s);
  for (int i = 0; i < len; ++i) {
    if (s[i] == 'h')
      for (int j = i + 1; j < len; ++j) {
        if (s[j] == 'e')
          for (int k = j + 1; k < len; ++k) {
            if (s[k] == 'l')
              for (int a = k + 1; a < len; ++a) {
                if (s[a] == 'l')
                  for (int b = a + 1; b < len; ++b) {
                    if (s[b] == 'o') {
                      puts("YES");
                      return 0;
                    }
                  }
              }
          }
      }
  }
  puts("NO");
  return 0;
}
