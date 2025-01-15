//
//  546A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>

using namespace std;

int main() {
  int k, n, w, s = 0;

  cin >> k >> n >> w;

  while (w) {
    s += w * k;
    w--;
  }

  s - n >= 0 ? cout << s - n : cout << "0";

  return 0;
}
