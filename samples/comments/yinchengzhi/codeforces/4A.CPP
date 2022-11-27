
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
using namespace std;
int main() {
  int n;
  while (scanf("%d", &n) != EOF) {
    if (n <= 2)
      printf("NO\n");
    else if (n % 2 == 0)
      printf("YES\n");
    else
      printf("NO\n");
  }
  return 0;
}
