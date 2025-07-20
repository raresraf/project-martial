// https://codeforces.com/problemset/problem/50/A

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
  int n, m, d;
  cin >> n >> m;
  int mul = n * m;
  if (mul % 2 == 0)
    cout << (mul / 2);

  else
    cout << ((mul - 1) / 2);
  // your code goes here
  return 0;
}