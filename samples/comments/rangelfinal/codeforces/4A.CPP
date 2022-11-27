#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  int w;

  cin >> w;
  cout << (w == 2 ? "NO" : (w % 2) == 0 ? "YES" : "NO");
  return 0;
}
