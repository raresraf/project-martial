#include <stdio.h>

using namespace std;

int main() {
  int x = 0, tc;
  char input[150];
  scanf("%d", &tc);
  while (tc--) {
    scanf("%s", input);
    if (input[1] == '+')
      ++x;
    else
      --x;
  }
  printf("%d\n", x);
  return 0;
}