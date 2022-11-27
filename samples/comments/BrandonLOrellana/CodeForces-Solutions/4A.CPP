/*
 *Link: https://codeforces.com/problemset/problem/4/A
 */

#include <iostream>
using namespace std;

int main() {
  unsigned int w;
  cin >> w;
  cout << (w % 2 == 0 && w > 2 ? "YES" : "NO") << endl;
}