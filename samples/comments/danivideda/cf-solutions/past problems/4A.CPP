#include <iostream>
using namespace std;

int main() {
  int x;
  cin >> x;

  if (x == 2) {
    cout << "NO" << endl;
  } else {
    if (x % 2 == 0) {
      cout << "genap" << endl;
    } else {
      cout << "ganjil" << endl;
    }
  }
  return 0;
}