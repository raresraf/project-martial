//
//  69A.cpp
//  codeforces
//
//  Created by Thiago Ribeiro on 6/27/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>

using namespace std;

int main() {
  int n, x = 0, y = 0, z = 0, a, b, c;

  cin >> n;

  while (n--) {
    scanf("%d %d %d", &a, &b, &c);
    x += a;
    y += b;
    z += c;
  }

  (x == 0 && y == 0 && z == 0) ? cout << "YES" << endl : cout << "NO" << endl;

  return 0;
}
