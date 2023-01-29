#include <iostream>

using namespace std;

int main() {
  int a;
  // read a.
  cin >> a;
  if (a % 2 == 0)
    // Check if a is even.
    cout << "even";
  else
    // Is odd.
    cout << "odd";

  return 0;
}
