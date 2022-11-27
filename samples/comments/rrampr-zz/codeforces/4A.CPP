#include <iostream>
using std::cin;
using std::cout;
int main() {
  int num;
  cin >> num;
  if (num % 2 || num == 2) {
    cout << "NO";
  } else {
    cout << "YES";
  }
  return 0;
}
