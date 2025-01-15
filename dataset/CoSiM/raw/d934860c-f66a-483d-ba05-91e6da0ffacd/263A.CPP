#include <cmath> //abs
#include <iostream>
#include <stdlib.h> //exit

using namespace std;

int main() {
  int n, x, y, num;
  for (int i = 1; i <= 25; i++) {
    cin >> num;
    if ((num) == 1) {
      x = ((i % 5) == 0) ? 5 : (i % 5);
      y = ((i % 5) == 0) ? (i / 5) : ((i / 5) + 1);
      break;
    }
  }
  // cout<<x<<y<<"\n";
  cout << (abs(x - 3) + abs(y - 3));
}
