#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>
using namespace std;
int main() {
  int up, down, count = 0;
  int i, j, a, sum = 0;
  cin >> a;
  for (i = 1; i <= a; i++) {
    cin >> down >> up;
    up = up - down;
    sum = sum + up;
    if (sum > count) {
      count = sum;
    }
  }
  cout << count << endl;

  return 0;
}
