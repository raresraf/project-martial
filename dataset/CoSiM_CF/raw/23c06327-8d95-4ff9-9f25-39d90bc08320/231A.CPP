//
//  231A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>
#include <stdio.h>

using namespace std;

int main() {
  int n, x, y, z, sum = 0, num = 0;

  cin >> n;

  while (n--) {
    scanf("%d %d %d", &x, &y, &z);
    sum += x + y + z;
    if (sum >= 2)
      num++;
    sum = 0;
  }

  cout << num << endl;

  return 0;
}
