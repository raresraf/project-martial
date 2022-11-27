#include <cstdio>

int main() {
  int num;
  scanf("%d", &num);
  if ((num & 1) || num < 3) {
    puts("NO");
  } else {
    puts("YES");
  }
}
