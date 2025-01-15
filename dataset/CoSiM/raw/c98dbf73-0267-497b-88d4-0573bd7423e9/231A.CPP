#include <cstdio>

using namespace std;

int main() {
  int n;
  int count = 0;
  scanf("%d", &n);
  for (int i = n; i > 0; i--) {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    if (a + b + c > 1) {
      count++;
    }
  }
  printf("%d\n", count);
}