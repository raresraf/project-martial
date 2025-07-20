#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int a, b, c;
  cin >> a >> b >> c;

  int case1 = a + b * c;
  int case2 = a * (b + c);
  int case3 = a * b * c;
  int case4 = (a + b) * c;

  if (case1 > case2 && case1 > case3 && case1 > case4) {
    cout << case1;
  } else if (case2 > case1 && case2 > case3 && case2 > case4) {
    cout << case2;
  } else if (case3 > case1 && case3 > case2 && case3 > case4) {
    cout << case3;
  } else {
    cout << case4;
  }

  return 0;
}