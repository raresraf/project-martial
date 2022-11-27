//
// Created by Conor on 5/27/2020.
//

#include <iostream>
using namespace std;

int main() {
  int num;
  cin >> num;

  if (num % 2 == 0 && num > 2) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
