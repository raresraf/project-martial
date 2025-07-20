//
// Created by Conor on 5/28/2020.
//

#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  int current = 0;
  int max = 0;
  int a, b;
  for (int i = 0; i < n; ++i) {
    cin >> a >> b;
    current -= a;
    current += b;

    if (current > max) {
      max = current;
    }
  }

  cout << max;

  return 0;
}