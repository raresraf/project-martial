#include <iostream>

using namespace std;

int main() {
  int input;
  cin >> input;
  if (input % 2 == 0 && input > 2) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
