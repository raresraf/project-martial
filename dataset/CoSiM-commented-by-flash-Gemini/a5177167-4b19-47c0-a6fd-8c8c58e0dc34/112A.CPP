#include <cctype>
#include <cstdio>
#include <cstring>

using namespace std;

int main() {
  char x[101], y[101];
  scanf("%s %s", &x, &y);
  for (int i = 0; x[i] != 0; i++) {
    x[i] = tolower(x[i]);
    y[i] = tolower(y[i]);
  }
  printf("%d", strcmp(x, y));
  return 0;
}