//
//  main.cpp
//  69A
//
//  Created by Alejandro Gleason on 3/21/19.
//  Copyright   2019 Alejandro Gleason. All rights reserved.
//

#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n; // lines
  int x, y, z, sx = 0, sy = 0, sz = 0;
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z; // scanning line by line
    sx = sx + x;
    sy = sy + y;
    sz = sz + z;
  }
  if (sx == 0 && sy == 0 && sz == 0) { // balanced
    printf("YES\n");
  } else if (sx != 0 || sy != 0 || sz != 0) { // non balanced
    printf("NO\n");
  }
  return 0;
}
