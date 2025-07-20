//
// Created by Conor on 5/27/2020.
//

#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  int a, b, c, count;
  count = 0;
  for (int i = 0; i < n; ++i) {
    cin >> a >> b >> c;
    if (a + b + c > 1) {
      count++;
    }
  }

  cout << count;
}