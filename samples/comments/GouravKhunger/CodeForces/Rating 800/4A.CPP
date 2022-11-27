// https://codeforces.com/problemset/problem/4/A
// 4A. Watermelon
// File Creation Date: 15-Oct-2020
// Author: Gourav(https://github.com/GouravKhunger)

#include <iostream>
using namespace std;

bool calculate(int n) {
  if (n % 2 == 0 && n != 2) {
    return true;
  } else {
    return false;
  }
}

int main() {
  ios::sync_with_stdio(false);
  int n;
  cin >> n;
  if (calculate(n)) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
