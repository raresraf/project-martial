#include <bits/stdc++.h>
#include <iostream>

int main() {
  int lines;
  std::cin >> lines;
  int x[lines], y[lines], z[lines];
  int sumx = 0, sumy = 0, sumz = 0;

  for (int i = 0; i < lines; i++) {
    std::cin >> x[i] >> y[i] >> z[i];
  }

  for (int i = 0; i < lines; i++) {
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
  }

  if (sumx == 0 && sumy == 0 && sumz == 0) {
    std::cout << "YES";
  } else {
    std::cout << "NO";
  }

  return 0;
}