//
//  96A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>

using namespace std;

int main() {
  string s, res = "NO";
  int countZeros = 0, countOnes = 0;

  cin >> s;

  for (const char &c : s) {
    if (c == '0') {
      countZeros++;
      countOnes = 0;
    } else {
      countOnes++;
      countZeros = 0;
    }

    if (countZeros >= 7) {
      countZeros = 0;
      res = "YES";
    }
    if (countOnes >= 7) {
      countOnes = 0;
      res = "YES";
    }
  }

  cout << res << "\n";

  // 100001000000110101100000

  return 0;
}
