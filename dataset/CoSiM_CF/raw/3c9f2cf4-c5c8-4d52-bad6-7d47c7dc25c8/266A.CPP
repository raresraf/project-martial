//
//  266A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>
#include <stdio.h>

using namespace std;

int main() {
  int n, count = 0;
  char carac = '\0', c;

  scanf("%d", &n);

  for (int i = 0; i < n; i++) {
    // scanf("%c", &c);
    cin >> c;
    if (i == 0) {
      carac = c;
    } else {
      if (carac == c) {
        count++;
      } else {
        carac = c;
        // count = 0;
      }
    }
  }

  printf("%d\n", count);
}
