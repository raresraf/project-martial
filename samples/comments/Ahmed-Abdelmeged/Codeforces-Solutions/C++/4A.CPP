#include <cmath>
#include <iostream>
using namespace std;

int main() {
  // the weight of the watermelon
  int w = 0;
  int halfOne = 2;
  // take input from the user
  cin >> w;
  w = w - halfOne;
  // print the result
  if (w % 2 == 0 && w != 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
