#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  // your code goes here
  int a[50][50];

  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> a[i][j];
    }
  }

  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {

      if (a[i][j] == 1)
        cout << (abs(3 - i) + abs(3 - j));
    }
  }
  // your code goes here
  return 0;
}