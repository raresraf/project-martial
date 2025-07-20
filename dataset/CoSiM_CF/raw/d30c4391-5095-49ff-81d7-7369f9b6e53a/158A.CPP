#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int i, n, k, peserta[51] = {0}, go = 0;
  cin >> n >> k;
  for (i = 1; i <= n; i++) {
    scanf("%d", &peserta[i]);
    if (peserta[i] >= peserta[k] && peserta[i] > 0)
      go++;
  }
  printf("%d", go);
}
