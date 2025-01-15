#include <cctype>
#include <cstdio>

using namespace std;

int main() {
  char fs[101], sd[101];
  char c, d;
  int suma = 0, sumb = 0;

  fscanf(stdin, "%s%s", fs, sd);

  for (int i = 0; fs[i] != '\0'; ++i) {
    c = tolower(fs[i]);
    d = tolower(sd[i]);

    if (c - '0' > d - '0') {
      fprintf(stdout, "1");
      return 0;
    } else if (c - '0' < d - '0') {
      fprintf(stdout, "-1");
      return 0;
    }
  }
  fprintf(stdout, "0");
  return 0;
}
