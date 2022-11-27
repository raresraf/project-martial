/*
Link: https://codeforces.com/problemset/problem/4/A
@uthor: chavodon
*/

#include <iostream>

using namespace std;

int main() {
  int w;
  cin >> w;
  if (w % 2 != 0 || w == 2)
    cout << "NO";
  else
    cout << "YES";
  return 0;
}
