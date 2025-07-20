#include <iostream>
using namespace std;
int main() {
  char c;
  c = getchar();
  while (c != '\n') {
    if (c == 'A' || c == 'a' || c == 'O' || c == 'o' || c == 'Y' || c == 'y' ||
        c == 'E' || c == 'e' || c == 'U' || c == 'u' || c == 'I' || c == 'i') {
      c = getchar();
      continue;
    } else {
      printf(".");
      if (c >= 'A' && c <= 'Z')
        printf("%c", c + 32);
      else
        printf("%c", c);
    }
    c = getchar();
  }
}