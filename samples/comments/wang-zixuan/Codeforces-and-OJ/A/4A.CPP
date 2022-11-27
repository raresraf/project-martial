#include <iostream>
using namespace std;

int main() {
  int kilo;
  cin >> kilo;
  if (kilo % 2 == 0 && kilo != 2)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}