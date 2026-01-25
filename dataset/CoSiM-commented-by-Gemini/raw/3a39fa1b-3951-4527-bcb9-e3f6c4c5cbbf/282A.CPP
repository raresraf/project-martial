#include <iostream>

using std::cin;
using std::cout;

int main(void) {
  int n;

  cin >> n;

  int count = 0;
  int value = 0;
  while (count++ < n) {
    char statement[] = {0, 0, 0, 0};
    cin >> statement;
    if (statement[0] == '+' || statement[2] == '+') {
      value++;
    } else if (statement[0] == '-' || statement[2] == '-') {
      value--;
    }
  }
  cout << value;

  return 0;
}