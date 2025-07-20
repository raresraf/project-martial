//
//  160A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, a, count = 0;
  vector<int> v;

  cin >> n;

  while (n--) {
    cin >> a;
    v.push_back(a);
  }

  sort(v.begin(), v.end());

  int sum = 0;
  for (const int i : v)
    sum += i;

  int x = 0;
  for (int i = (int)v.size() - 1; i >= 0; i--) {
    count++;
    x += v[i];
    sum -= v[i];
    if (x > sum) {
      break;
    }
  }

  cout << count << endl;

  return 0;
}
