/*
 *Link: https://codeforces.com/problemset/problem/263/A
 */

#include <iostream>
using namespace std;

unsigned int diff(int a, int b) {
  int res = a - b;
  if (res < 0)
    return res * -1;
  else
    return res;
}

int main() {
  unsigned int num, position;
  for (int i = 0; i < 25; i++) {
    cin >> num;
    if (num == 1)
      position = i;
  }
  int res = diff(2, position / 5) + diff(2, position % 5);
  cout << res << endl;
}