/*
Link: https://codeforces.com/problemset/problem/231/A
@uthor: chavodon
*/

#include <iostream>

using namespace std;

int main() {
  int n, i, n1, n2, n3, sum = 0;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> n1 >> n2 >> n3;
    if ((n1 + n2 + n3) >= 2)
      sum++;
  }
  cout << sum;
  return 0;
}