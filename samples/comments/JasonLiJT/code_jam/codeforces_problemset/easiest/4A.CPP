#include <cstdio>

using namespace std;

int main() {
  int w;
  scanf("%d", &w);
  if (w % 2 == 0 && w > 3)
    printf("YES\n");
  else
    printf("NO\n");
}