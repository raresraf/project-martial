#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int n, x = 0;
  char a, b, c;
  scanf("%d%*c", &n); // %*c means ignoring a character
  for (int i = 0; i < n; ++i) {
    scanf("%c%c%c%*c", &a, &b, &c); // %*c means ignoring a character
    if (b == '+')
      ++x;
    else
      --x;
  }

  printf("%d\n", x);
}