#include <iostream>

using namespace std;

int main() {
  string palabra;
  cin >> palabra;
  if (palabra[0] > 96)
    palabra[0] -= 32;
  cout << palabra;
  return 0;
}
