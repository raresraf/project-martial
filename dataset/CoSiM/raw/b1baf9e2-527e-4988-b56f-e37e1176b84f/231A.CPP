// https://codeforces.com/problemset/problem/231/A

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

using namespace std;

int main() {
  // your code goes here
  int n;
  cin >> n;
  int sum = 0, d = 0;
  int a[1002][3];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1)
        sum++;
    }
    if (sum > 1)
      d++;
    sum = 0;
  }

  cout << d;
  // your code goes here
  return 0;
}