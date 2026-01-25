#include <cctype>
#include <cstdio>

using namespace std;

int main() {
  char x[1001];
  scanf("%s", &x);
  x[0] = toupper(x[0]);
  printf("%s", x);
  return 0;
}