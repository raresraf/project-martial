#include <cstdio>

using namespace std;

int main() {
  char x[101];
  int count = 1;
  scanf("%s", &x);
  for (int i = 0; x[i + 1] != 0; i++) {
    if (x[i + 1] != x[i]) {
      count = 1;
    } else {
      count++;
    }
    if (count >= 7) {
      printf("YES");
      return 0;
    }
    // printf("%d\n",count);
  }
  printf("NO");
  return 0;
}