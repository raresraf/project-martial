// Sagindyk Urazayev
// codeforces.com/problemset/problem/4/A
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  if ((n < 1) || (n > 100)) {
    cout << "Wrong value!" << endl;
  }
  if ((n % 2 != 0) || (n == 2)) {
    cout << "NO\n";
  } else {
    cout << "YES\n";
  }
  return 0;
}
