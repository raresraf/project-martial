//
//  4A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int weigh;

  scanf("%d", &weigh);

  if (weigh == 1 || weigh == 2)
    printf("NO");
  else if (weigh % 2 == 0)
    printf("YES");
  else
    printf("NO");

  return 0;
}
