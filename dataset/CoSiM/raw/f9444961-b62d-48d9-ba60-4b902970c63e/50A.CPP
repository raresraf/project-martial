#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  int m, n;

  cin >> m >> n;

  cout << (((m * n % 2) == 0) ? m * n / 2 : (m * n - 1) / 2);

  return 0;
}
