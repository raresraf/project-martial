#include <cmath>
#include <stdio.h>
using namespace std;

int main() {

  int num;
  for (int i = 1; i < 6; i++) {
    for (int j = 1; j < 6; j++) {
      scanf("%d", &num);
      if (num) {
        printf("%d", abs(i - 3) + abs(j - 3));
        i = 6;
      }
    }
  }

  return 0;
}
