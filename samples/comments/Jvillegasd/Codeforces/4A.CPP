#include <stdio.h>

using namespace std;

int main() {
  int w;
  scanf("%d", &w);
  if (w % 2 != 0 || w == 2)
    printf("NO\n");
  else
    printf("YES\n");
  return 0;
}