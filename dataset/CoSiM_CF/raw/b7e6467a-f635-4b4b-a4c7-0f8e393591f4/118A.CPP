#include <cctype>
#include <cstdio>

using namespace std;

int main() {
  char s[100];
  char a[13] = "AOYEUIaoyeui";
  int found = 0;
  char c;

  fscanf(stdin, "%s", s);

  for (int i = 0; s[i] != '\0'; ++i) {
    for (int j = 0; a[j] != '\0'; ++j) {
      if (s[i] == a[j]) {
        found = 1;
        break;
      }
    }
    if (!found) {
      c = s[i];
      fprintf(stdout, ".%c", tolower(c));
    }
    found = 0;
  }
}
