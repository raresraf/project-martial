// CodeForces
// Problem 4A [Watermelon]

#include <iostream>
using namespace std;

int main() {
  int w;
  cin >> w;
  (w >= 4 && w % 2 == 0) ? cout << "YES" : cout << "NO";
  return 0;
}