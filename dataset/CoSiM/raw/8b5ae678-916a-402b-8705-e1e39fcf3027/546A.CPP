#include <cstdio>

using namespace std;

int main() {
  int count, row, money, cost;
  scanf("%d %d %d", &cost, &money, &row);
  if (row == 1) {
    if (money > cost)
      printf("0");
    else
      printf("%d", cost - money);
  } else {
    count = (row * (row + 1)) / 2;
    count = count * cost;
    if (money > count)
      printf("0");
    else
      printf("%d", count - money);
  }
  return 0;
}
