#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  int n;

  cin >> n;

  int counter = 0;

  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    if ((a + b + c) >= 2)
      counter++;
  }

  cout << counter;
  return 0;
}
