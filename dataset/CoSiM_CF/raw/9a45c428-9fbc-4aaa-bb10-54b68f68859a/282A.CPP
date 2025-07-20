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

  int valor = 0;

  for (int i = 0; i < n; i++) {
    string line;
    cin >> line;
    if (line[0] == '-' || line[2] == '-')
      valor--;
    else
      valor++;
  }

  cout << valor;

  return 0;
}
