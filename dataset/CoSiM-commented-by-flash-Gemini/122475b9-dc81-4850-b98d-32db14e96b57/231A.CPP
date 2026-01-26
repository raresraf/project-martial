#include <iostream>
using namespace std;
int main() {
  int testcases;
  cin >> testcases;
  int Petya, Vasya, Tonya, solutions = 0;
  while (testcases > 0) {
    cin >> Petya >> Vasya >> Tonya;
    if ((Petya + Vasya + Tonya) >= 2) {
      solutions++;
    }
    testcases--;
  }
  cout << solutions;
  return 0;
}
