//
//  158A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int n;
  int k;
  int pt[50];
  int count = 0;

  scanf("%d %d", &n, &k);

  for (int i = 0; i < n; i++)
    scanf("%d", &pt[i]);
  for (int i = 0; i < n; i++) {
    if (pt[0] == 0) {
      if (pt[i + 1] == 0)
        count = 0;
    } else if (pt[i] >= pt[k - 1] && pt[i] != 0)
      count++;
  }

  printf("%d\n", count);

  return 0;
}
