#include <cstdio>

using namespace std;

int main() {
  int T, A = 0, B = 0, C = 0, temp;
  scanf("%d", &T);

  while (T--) {
    scanf("%d", &temp);
    A += temp;
    scanf("%d", &temp);
    B += temp;
    scanf("%d", &temp);
    C += temp;
  }

  if (A == 0 && B == 0 && C == 0)
    printf("YES\n");
  else
    printf("NO\n");

  return 0;
}