#include <cctype>
#include <cstdio>

using namespace std;

int main() {
  char x[101];
  scanf("%s", &x);
  for (int i = 0; x[i] != 0; i++) {
    x[i] = tolower(x[i]);
    if (x[i] != 'a' && x[i] != 'e' && x[i] != 'i' && x[i] != 'o' &&
        x[i] != 'u' && x[i] != 'y') {
      printf(".%c", x[i]);
    }
  }
  return 0;
}