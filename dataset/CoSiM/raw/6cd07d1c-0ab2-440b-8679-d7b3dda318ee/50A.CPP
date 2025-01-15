#include <iostream>
using namespace std;
int main(int argc, char *argv[]) {
  int n, m;
  scanf("%d %d", &n, &m);
  printf("%d\n", (n / 2) * m + (n % 2) * (m / 2));
  return 0;
}
