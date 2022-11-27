// CodeForces
// A.Watermelon
// GNU C++17

#include <iostream>
using namespace std;

int main() {
  int watermelonSize;
  cin >> watermelonSize;
  if (watermelonSize % 2 == 0 && watermelonSize > 2)
    cout << "YES" << endl;
  else {
    cout << "No" << endl;
  }
}