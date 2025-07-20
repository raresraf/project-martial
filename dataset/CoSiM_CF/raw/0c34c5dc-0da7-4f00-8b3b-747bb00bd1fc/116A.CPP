//
//  116A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <stdio.h>

int main() {
  int a, b, v = 0, m = 0, n;

  scanf("%d", &n);

  while (n--) {
    scanf("%d %d", &a, &b);
    v += b - a;
    if (v > m)
      m = v;
  }

  printf("%d\n", m);

  return 0;
}
