// https://codeforces.com/problemset/problem/546/A

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
  int k, n, w;
  cin >> k >> n >> w;

  int sum = 0;
  for (int i = 1; i <= w; i++) {
    sum += i * k;
  }
  if (sum > n)
    cout << sum - n;

  else
    cout << "0";
  // your code goes here
  return 0;
}