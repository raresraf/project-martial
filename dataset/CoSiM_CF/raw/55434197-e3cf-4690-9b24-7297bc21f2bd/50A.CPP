//
// Created by Conor on 5/27/2020.
//

#include <iostream>
using namespace std;

int main() {
  int m, n, count;
  cin >> m >> n;

  count = n * (m / 2);
  if (m % 2 == 1) {
    count += n / 2;
  }
  cout << count;

  return 0;
}