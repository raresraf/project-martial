#include <iostream>

using namespace std;

int main() {
  int a;
  // read a.
  cin >> a;
  if (a % 2 == 0)
    // Check even.
    cout << "even";
  else
    // Check odd.
    cout << "odd";
  return 0;
}
