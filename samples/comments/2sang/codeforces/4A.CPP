#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int w;
  scanf("%d", &w);
  if (w == 2)
    printf("NO\n");
  else if (w % 2 == 0)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
