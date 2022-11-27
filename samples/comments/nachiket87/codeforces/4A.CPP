#include <iostream>

using namespace std;

int main() {
  int a;
  cin >> a;
  if (a <= 2) {
    cout << "NO" << endl;
  } else {
    if (a % 2 == 0) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
  return 0;
}