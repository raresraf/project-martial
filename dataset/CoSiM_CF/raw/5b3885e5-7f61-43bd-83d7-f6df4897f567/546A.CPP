#include <cstdio>

using namespace std;

int main() {
  int k, n, w;
  int amount;
  scanf("%d %d %d", &k, &n, &w);
  amount = (w * (w + 1) * k / 2) - n;
  printf("%d", amount > 0 ? amount : 0);
}